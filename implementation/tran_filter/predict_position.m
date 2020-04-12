function [im_patch_cf, response_cf_all, response_cf] = predict_position(im, pos, p, bg_area, features, global_feat_params, feat_type, layerInd, hann_window, model_x_f, model_w_f)

im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);         % lfl: 抠图
z  = cell(2,1);
z_f = cell(2,1);
kz_f = cell(2,1);

for M = 1:2
    z{M} = bsxfun(@times, get_features(im_patch_cf, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
    z_f{M} = double(fft2(z{M}));
    switch p.kernel_type
        case 'gaussian'
            kz_f{M} = gaussian_correlation(z_f{M}, model_x_f{M}, p.tran_sigma{M});
        case 'polynomial'
            kz_f{M} = polynomial_correlation(z_f{M}, model_x_f{M}, p.polya{M}, p.polyb{M});
        case 'linear'
            kz_f{M} = sum(z_f{M} .* conj(model_x_f{M}), 3) / numel(z_f{M});
    end
end

    response_cf{1} = real(ifft2(model_w_f{1} .* kz_f{1}));
    response_cf{2} = real(ifft2(model_w_f{2} .* kz_f{2}));

    % Crop square search region (in feature pixels).
    response_cf{1} = cropFilterResponse(response_cf{1}, ... % 这里面已经经历了fftshift
        floor_odd(p.norm_delta_area / p.hog_cell_size));
    response_cf{2} = cropFilterResponse(response_cf{2}, ...
        floor_odd(p.norm_delta_area / p.hog_cell_size));

    if p.hog_cell_size > 1
        % Scale up to match center likelihood resolution.
        response_cf{1} = mexResize(response_cf{1}, p.norm_delta_area,'auto');
        response_cf{2} = mexResize(response_cf{2}, p.norm_delta_area,'auto');
    end

    p1 = adaptive_weight(response_cf{1});
    p2 = adaptive_weight(response_cf{2});

    response_cf_all = (p1.*response_cf{1}./max(response_cf{1}(:))) + (p2 .* response_cf{2}./max(response_cf{2}(:)));

