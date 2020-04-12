function results = tracker(p, im, bg_area, fg_area, area_resize_factor)

%% key parameters
two_feature = p.two_feature;
if two_feature
    feat_num = 2;
    feat_dim{1} = p.feat1dim;
    feat_dim{2} = p.feat2dim;
else
    feat_num = 1;
    feat_dim{1} = p.feat1dim;
end
dim_attention_avail = p.channel_attention_avail;
dim_act_threshold = p.dim_act_threshold;
if dim_attention_avail
    dim_wlr = p.chann_wlr;
end
position_attention_avail = p.position_attention_avail;
if position_attention_avail
    pos_lr = p.pos_lr;
end

%% Intrinsic parameters
feat_type = p.feat_type;
layerInd = p.layerInd;

lambda1 = p.lambda1;
lambda2 = p.lambda2;
features = p.t_features;
global_feat_params = p.t_global;
num_frames = numel(p.img_files);
s_frames = p.s_frames;
video_path = p.video_path;
% used for benchmark
rect_positions = zeros(num_frames, 4);
pos = p.init_pos;
target_sz = p.target_sz;
hann_window_cosine = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');

offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2)];
output_sigma{1} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{1} / p.hog_cell_size;
y{1} = gaussianResponse(p.cf_response_size, output_sigma{1});
yf{1} = fft2(y{1});
if two_feature
    output_sigma{2} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{2} / p.hog_cell_size;
    y{2} = gaussianResponse(p.cf_response_size, output_sigma{2});
    yf{2} = fft2(y{2});
end

% variables initialization
model_x_f = cell(2,1);
model_w_f = cell(2,1);
z  = cell(2,1);
z_f = cell(2,1);
kz_f = cell(2,1);
x = cell(2,1);
x_f = cell(2,1);
k_f = cell(2,1);
learning_rate_pwp = p.learning_rate_pwp;
% patch of the target + padding
patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;

%% Scale filter
% from DSST
p.num_scales = 33;
p.hog_scale_cell_size = 4;
p.learning_rate_scale = 0.025;
p.scale_sigma_factor = 1/2;
p.scale_model_factor = 1.0;
p.scale_step = 1.03;
p.scale_model_max_area = 32*16;
p.lambda = 1e-4;

scale_factor = 1;
scale_base_target_sz = target_sz;
scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
ss = (1:p.num_scales) - ceil(p.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(p.num_scales,2) == 0
    scale_window = single(hann(p.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(p.num_scales));
end
ss = 1:p.num_scales;
scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
    p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
end
scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
% find maximum and minimum scales
min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

% initialization
trans_vec = [];

%% MAIN LOOP
t_imread = 0;
tic;
for frame = 1:num_frames
    if frame > 1
        tic_imread = tic;
        % Load the image at the current frame
        im = imread([s_frames{frame}]);
        t_imread = t_imread + toc(tic_imread);
        
        %% TESTING step
        im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        likelihood_map = getColourMap(im_patch_cf, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        likelihood_map(isnan(likelihood_map)) = 0;
        likelihood_map = imResample(likelihood_map, p.cf_response_size);
        likelihood_map = (likelihood_map + min(likelihood_map(:)))/(max(likelihood_map(:)) + min(likelihood_map(:)));
        if (sum(likelihood_map(:))/prod(p.cf_response_size)<0.01)
            likelihood_map = 1;
        end
        likelihood_map = max(likelihood_map, 0.1);
        hann_window =  hann_window_cosine .* likelihood_map;
        if ~dim_attention_avail
            for M = 1:feat_num
                z{M} = bsxfun(@times, get_features(im_patch_cf, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
                z_f{M} = double(fft2(z{M}));
                switch p.kernel_type{M}
                    case 'gaussian'
                        kz_f{M} = gaussian_correlation(z_f{M}, model_x_f{M}, p.tran_sigma{M});
                    case 'polynomial'
                        kz_f{M} = polynomial_correlation(z_f{M}, model_x_f{M}, p.polya{M}, p.polyb{M});
                    case 'linear'
                        kz_f{M} = sum(z_f{M} .* conj(model_x_f{M}), 3) / numel(z_f{M});
                end
            end
            response_cf{1} = real(ifft2(model_w_f{1} .* kz_f{1}));
            % Crop square search region (in feature pixels).
            response_cf{1} = cropFilterResponse(response_cf{1},floor_odd(p.norm_delta_area / p.hog_cell_size));
            if two_feature
                response_cf{2} = real(ifft2(model_w_f{2} .* kz_f{2}));
                response_cf{2} = cropFilterResponse(response_cf{2},floor_odd(p.norm_delta_area / p.hog_cell_size));
            end
        else
            shift_hann = circshift(hann_window_cosine, -floor(p.cf_response_size(1:2)/2));
            for M = 1:feat_num
                z{M} = bsxfun(@times, get_features(im_patch_cf, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
                z_f{M} = double(fft2(z{M}));
                size_zf = size(z_f{M});
                switch p.kernel_type{M}
                    case 'gaussian'
                        for d = 1:feat_dim{M}
                            kz_f{M}(:,:,d) = gaussian_correlation(z_f{M}(:,:,d), model_x_f{M}(:,:,d), p.tran_sigma{M});
                        end
                    case 'polynomial'
                        for d = 1:feat_dim{M}
                            kz_f{M}(:,:,d) = polynomial_correlation(z_f{M}(:,:,d), model_x_f{M}(:,:,d), p.polya{M}, p.polyb{M});
                        end
                    case 'linear'
                        kz_f{M} = bsxfun(@times, z_f{M}, conj(model_x_f{M})) / prod(size_zf(1:2));
                end
            end
            for M = 1:feat_num
                response_cf_chann{M} = real(ifft2(model_w_f{M} .* kz_f{M}));
                response_cf_ca{M}    = bsxfun(@times, response_cf_chann{M}, reshape(model_chann_w{M}, 1, 1, size(response_cf_chann{M},3)));
                if position_attention_avail
                    pa_map{M}   = pos_attention(response_cf_ca{M}, hann_window, frame, target_sz, trans_vec);
                    if mod(frame,25) == 1
                        testest = 1;
                    end
                    if frame == 2
                        model_pa_map{M}  = pa_map{M};
                    else
                        model_pa_map{M} = pos_lr * pa_map{M} + (1-pos_lr) * model_pa_map{M};
                    end
                    response_cf_cpa{M} = bsxfun(@times, response_cf_ca{M}, model_pa_map{M});
                    response_cf{M}       = sum(response_cf_cpa{M}, 3);
                else
                    response_cf{M}       = sum(response_cf_ca{M}, 3);
                end
                % Crop square search region (in feature pixels).
                response_cf{M} = cropFilterResponse(response_cf{M},floor_odd(p.norm_delta_area / p.hog_cell_size));
            end
        end
        
        if p.context_att
            if mod(frame - 1, p.bgl_interv) == 0
                gPMER(frame-1) = calPMER(response_cf{1});
                feat_order = 1;
                z_bf{feat_order} = zeros([size(x_f{feat_order}) length(offset)], 'single');
                for j = 1:length(offset)
                    im_patch_bg = getSubwindow(im, pos+offset(j,:), p.norm_bg_area, bg_area);
                    z_b = bsxfun(@times, get_features(im_patch_bg, features, global_feat_params, feat_type{feat_order}, layerInd{feat_order}), hann_window_cosine);
                    z_bf{feat_order}(:,:,:,j) = fft2(z_b);
                    switch p.kernel_type{feat_order}
                        case 'gaussian'
                            kz_bf{feat_order}(:,:,j) = gaussian_correlation(z_bf{feat_order}(:,:,:,j), z_bf{feat_order}(:,:,:,j), p.tran_sigma{feat_order});
                        case 'polynomial'
                            kz_bf{feat_order}(:,:,j) = polynomial_correlation(z_bf{feat_order}(:,:,:,j), z_bf{feat_order}(:,:,:,j), p.polya{feat_order}, p.polyb{feat_order});
                        case 'linear'
                            kz_bf{feat_order} = bsxfun(@times, z_bf{feat_order} , conj(z_bf{feat_order})) / prod(size_zf(1:2));
                    end
                    response_context = real(ifft2(model_w_f{1} .* kz_bf{feat_order}(:,:,j)));
                    % Crop square search region (in feature pixels).
                    response_context = cropFilterResponse(response_context,floor_odd(p.norm_delta_area / p.hog_cell_size));
                    cPMER(frame-1,j) = calPMER(response_context);
                end
                part_i = cPMER(frame-1,:) / gPMER(frame-1);
                part_i = max(part_i, 0.3);
                % activation and normalization to 1
                penalty(frame,:) = 4 * part_i.^2 / sumsqr(part_i);
                model_p(frame,:) = 0.2 * penalty(frame,:) + 0.8 * model_p(frame-p.bgl_interv,:);
            end
        end
        
        % Scale up to match center likelihood resolution.
        response_cf{1} = mexResize(response_cf{1}, p.norm_delta_area,'auto');
        if two_feature
            response_cf{2} = mexResize(response_cf{2}, p.norm_delta_area,'auto');
            p1 = adaptive_weight(response_cf{1});
            p2 = adaptive_weight(response_cf{2});
            sum_p = p1 + p2;
            p1 = p1/sum_p; p2 = p2/sum_p;
            response_cf_all = ...
                (p1.*response_cf{1}./max(response_cf{1}(:))) + ...
                (p2.*response_cf{2}./max(response_cf{2}(:)));
        else
            response_cf_all = response_cf{1};
        end
        center = (1+p.norm_delta_area) / 2;
        response = response_cf_all;
        
        [row, col] = find(response == max(response(:)));
        row = row(1);
        col = col(1);
        delta_row = row - center(1);
        delta_col = col - center(2);
        %         sprintf("frame: %d", frame)
        trans_vec = ([delta_row, delta_col]) / area_resize_factor;
        pos = pos + trans_vec;
        %         pos = pos(1:2); % avoid more than one max points
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %% SCALE SPACE SEARCH
        im_patch_scale = getScaleSubwindow(im, pos, scale_base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        xsf = fft(im_patch_scale,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
        recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
        %set the scale
        scale_factor = scale_factor * scale_factors(recovered_scale);
        
        if scale_factor < min_scale_factor
            scale_factor = min_scale_factor;
        elseif scale_factor > max_scale_factor
            scale_factor = max_scale_factor;
        end
        % use new scale to update bboxes for target, filter, bg and fg models
        target_sz = round(scale_base_target_sz * scale_factor);
        p.avg_dim = sum(target_sz)/2;
        bg_area = round(target_sz + p.avg_dim * p.padding);
        if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
        if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end
        
        bg_area = bg_area - mod(bg_area - target_sz, 2);
        fg_area = round(target_sz - p.avg_dim * p.inner_padding);
        fg_area = fg_area + mod(bg_area - fg_area, 2);
        % Compute the rectangle with (or close to) params.fixed_area and same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
    end
    
    %% Train and Update Model
    %% 中心样本 im_patch_fg
    obj = getSubwindow(im, pos, target_sz);
    im_patch_fg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    for M = 1:feat_num
        x{M} = bsxfun(@times, get_features(im_patch_fg, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window_cosine);
        x_f{M} = double(fft2(x{M}));
        size_xf = size(x_f{M});
        if dim_attention_avail
            switch p.kernel_type{M}
                case 'gaussian'
                    for d = 1:feat_dim{M}
                        k_f{M}(:,:,d) = gaussian_correlation(x_f{M}(:,:,d), x_f{M}(:,:,d), p.tran_sigma{M});
                    end
                case 'polynomial'
                    for d = 1:feat_dim{M}
                        k_f{M}(:,:,d) = polynomial_correlation(x_f{M}(:,:,d), x_f{M}(:,:,d), p.polya{M}, p.polyb{M});
                    end
                case 'linear'
                    k_f{M} = bsxfun(@times, x_f{M}, conj(x_f{M})) / prod(size_xf(1:2));
            end
        else
            switch p.kernel_type{M}
                case 'gaussian'
                    k_f{M} = gaussian_correlation(x_f{M}, x_f{M}, p.tran_sigma{M});
                case 'polynomial'
                    k_f{M} = polynomial_correlation(x_f{M}, x_f{M}, p.polya{M}, p.polyb{M});
                case 'linear'
                    k_f{M} = sum(x_f{M} .* conj(x_f{M}), 3) / numel(x_f{M});
            end
        end
    end
    
    %% contextual attention
    if p.context_att
        if frame == 1
            model_p(1,:) = [1 1 1 1];
        end
        if mod(frame - 1, p.bgl_interv) == 0
            ijk = 1;
            x_bf{ijk} = zeros([size(x_f{ijk}) length(offset)],'single');
            for j = 1:length(offset)
                context = getSubwindow(im, pos+offset(j,:), target_sz);
                im_patch_bg = getSubwindow(im, pos+offset(j,:), p.norm_bg_area, bg_area);
                x_b = bsxfun(@times, get_features(im_patch_bg, features, global_feat_params, feat_type{ijk}, layerInd{ijk}), hann_window_cosine);
                x_bf{ijk}(:,:,:,j) = fft2(x_b * model_p(frame,j) * sqrt(lambda2));
                switch p.kernel_type{ijk}
                    case 'gaussian'
                        for d = 1:feat_dim{ijk}
                            k_bf{ijk}(:,:,d,j) = gaussian_correlation(x_bf{ijk}(:,:,d,j), x_bf{ijk}(:,:,d,j), p.tran_sigma{ijk});
                        end
                    case 'polynomial'
                        for d = 1:feat_dim{ijk}
                            k_bf{ijk}(:,:,d,j) = polynomial_correlation(x_bf{ijk}(:,:,d,j), x_bf{ijk}(:,:,d,j), p.polya{ijk}, p.polyb{ijk});
                        end
                    case 'linear'
                        k_bf{ijk} = bsxfun(@times, x_bf{ijk}, conj(x_bf{ijk})) / prod(size_xf(1:2));
                end
            end
            new_wf_num{1} = k_f{1} .* yf{1};
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1 + sum(k_bf{1} .* conj(k_bf{1}), 4);
            if two_feature
                new_wf_num{2} = k_f{2} .* yf{2};
                new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1;
            end
        else
            new_wf_num{1} = k_f{1} .* yf{1};
            new_wf_den{1} = k_f{1} .* conj(k_f{1}) + lambda1;
            if two_feature
                new_wf_num{2} = k_f{2} .* yf{2};
                new_wf_den{2} = k_f{2} .* conj(k_f{2}) + lambda1;
            end
        end
    else
        for M = 1 : feat_num
            new_wf_num{M} = k_f{M} .* yf{M};
            new_wf_den{M} = k_f{M} .* conj(k_f{M}) + lambda1;
        end
    end
    for M = 1:feat_num
        w_f{M} = new_wf_num{M} ./ new_wf_den{M};
    end
    
    %% calculate per-channel feature weights
    if dim_attention_avail
        for M = 1:feat_num
            if frame == 1
                model_chann_w{M} = ones(1, feat_dim{M},'single');
            end
            response_lr{M} = real(ifft2(w_f{M} .* k_f{M}));
            chann_w{M} = (max(reshape(response_lr{M}, [size(response_lr{M},1)*size(response_lr{M},2), size(response_lr{M},3)]), [], 1) ...
                +mean(mean(reshape(response_lr{M}, [size(response_lr{M},1)*size(response_lr{M},2), size(response_lr{M},3)]))));
            chann_w{M} = feat_dim{M} * chann_w{M} / sum(chann_w{M});
            chann_w{M} = max(chann_w{M} - dim_act_threshold, 0) + dim_act_threshold;
            model_chann_w{M} = (1-dim_wlr)*model_chann_w{M} + dim_wlr*chann_w{M};
            model_chann_w{M} = feat_dim{M} * model_chann_w{M} / sum(model_chann_w{M});
        end
    end
    
    %% Initialization
    if frame == 1
        % first frame, train with a single image
        model_x_f = x_f;
        model_w_f = w_f;
    else
        % subsequent frames, update the model by linear interpolation
        for M = 1:feat_num
            model_x_f{M} = (1 - p.learning_rate_cf) * model_x_f{M} + p.learning_rate_cf * x_f{M};
            model_w_f{M} = (1 - p.learning_rate_cf) * model_w_f{M} + p.learning_rate_cf * w_f{M};
        end
        % BG/FG MODEL UPDATE   patch of the target + padding
        im_patch_color = getSubwindow(im, pos, p.norm_bg_area, bg_area*(1-p.inner_padding));
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_color, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, learning_rate_pwp);
    end
    %% Upadate Scale
    im_patch_scale = getScaleSubwindow(im, pos, scale_base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
    xsf = fft(im_patch_scale,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end
    %% update bbox position
    if (frame == 1)
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    end
    rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];
    rect_positions(frame,:) = rect_position;
    
    elapsed_time = toc;
    
    %% Visualization
    if p.visualization == 1
        if frame == 1   %first frame, create GUI
            figure('Name',['Tracker - ' video_path]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',2);
            rect_handle2 = rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','--', 'EdgeColor','b');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im);
                set(rect_handle, 'Position', rect_position);
                set(rect_handle2, 'Position', rect_position_padded);
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        drawnow
    end
end
results.model_p = model_p;

%% save data for benchmark
results.type = 'rect';
results.res = rect_positions;
results.fps = num_frames/(elapsed_time - t_imread);
fprintf('fps: %f',results.fps)

end