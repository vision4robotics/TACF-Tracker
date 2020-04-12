function [model_x_f, model_w_f,im_patch_bg] = update_model(frame, im, pos, p, bg_area, features, global_feat_params, feat_type, layerInd, hann_window, offset, model_x_f, model_w_f, yf, gamma, lambda1, lambda2)

im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
x = cell(2,1);
x_f = cell(2,1);
k_f = cell(2,1);
% lfl: 获得pos中心样本和context-aware区域的样本
for M = 1:2
    x{M} = bsxfun(@times, get_features(im_patch_bg, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
    x_f{M} = double(fft2(x{M}));
    switch p.kernel_type
        case 'gaussian'
            k_f{M} = gaussian_correlation(x_f{M}, x_f{M}, p.tran_sigma{M});
        case 'polynomial'
            k_f{M} = polynomial_correlation(x_f{M}, x_f{M}, p.polya{M}, p.polyb{M});
        case 'linear'
            k_f{M} = sum(x_f{M} .* conj(x_f{M}), 3) / numel(x_f{M});
    end
end

% 每xxx帧更新背景信息
if mod(frame, p.bgl_interv) == 0
    x_bf{1} = zeros([size(x_f{1}) length(offset)]);
    for j = 1:length(offset)
        %obtain a subwindow close to target for regression to 0
        context_patch = getSubwindow(im, pos+offset(j,:), p.norm_bg_area, bg_area);
        x_b = bsxfun(@times, get_features(context_patch, features, global_feat_params, feat_type{1}, layerInd{1}), hann_window);
        x_bf{1}(:,:,:,j) = fft2(x_b);      
        switch p.kernel_type
            case 'gaussian'
                k_bf{1}(:,:,j) = gaussian_correlation(x_bf{1}(:,:,:,j), x_bf{1}(:,:,:,j), p.tran_sigma{1});
            case 'polynomial'
                k_bf{1}(:,:,j) = polynomial_correlation(x_bf{1}(:,:,:,j), x_bf{1}(:,:,:,j), p.polya{1}, p.polyb{1});
            case 'linear'
                k_bf{1} = sum(x_bf{1}(:,:,:,j) .* conj(x_bf{1}(:,:,:,j)), 3) / numel(x_bf{1}(:,:,:,j));
        end
    end
    A{1} = (1 + gamma) .* k_f{1} + lambda1 + lambda2 * sum(conj(k_bf{1}) .* k_bf{1}, 3);
    A{2} = (1 + gamma) .* k_f{2} + lambda1;
else
    A{1} = (1 + gamma) .* k_f{1} + lambda1;
    A{2} = (1 + gamma) .* k_f{2} + lambda1;
end
if p.output_sigma_factor{1} == p.output_sigma_factor{2}
    new_wf_num{1} = yf{1};                              
    new_wf_den{1} = A{1} - ((A{1} + gamma .* k_f{1}) .* gamma .* k_f{2}) ./ (A{2} + gamma .* k_f{2});
    w_f{1} = new_wf_num{1} ./ new_wf_den{1};

    new_wf_num{2} = yf{2};
    new_wf_den{2} = A{2} - ((A{2} + gamma .* k_f{2}) .* gamma .* k_f{1}) ./ (A{1} + gamma .* k_f{1});                                            
    w_f{2} = new_wf_num{2} ./ new_wf_den{2};
else
    w_f{1} = (A{2} .* yf{1} + gamma .* yf{2} .* k_f{2}) ./ (A{1} .* A{2} - gamma^2 .* k_f{1} .* k_f{2});                          
    w_f{2} = (A{1} .* yf{2} + gamma .* yf{1} .* k_f{1}) ./ (A{2} .* A{1} - gamma^2 .* k_f{1} .* k_f{2});                           
end

if frame == 1
    % first frame, train with a single image
    model_x_f = x_f;
    model_w_f = w_f;
else
    % subsequent frames, update the model by linear interpolation
    model_x_f{1} = (1 - p.learning_rate_cf) * model_x_f{1} + p.learning_rate_cf * x_f{1};
    model_x_f{2} = (1 - p.learning_rate_cf) * model_x_f{2} + p.learning_rate_cf * x_f{2};
    model_w_f{1} = (1 - p.learning_rate_cf) * model_w_f{1} + p.learning_rate_cf * w_f{1};
    model_w_f{2} = (1 - p.learning_rate_cf) * model_w_f{2} + p.learning_rate_cf * w_f{2};
end