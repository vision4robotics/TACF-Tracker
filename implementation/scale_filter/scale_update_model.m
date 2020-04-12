function [im_patch_scale, model_hf] = scale_update_model(p, im, pos, base_target_sz, scale_factors, scale_factor, scale_window, scale_model_sz, frame, im_patch_scale, ysf)
if p.scale_adaptation
    im_patch_scale_frame = getScaleSubwindow_v1(im, pos, base_target_sz, scale_factors*scale_factor, scale_window, scale_model_sz, p.hog_scale_cell_size);

    if frame == 1
        im_patch_scale=im_patch_scale_frame;
    else
        im_patch_scale=(1 - p.learning_rate_scale)*im_patch_scale + p.learning_rate_scale*im_patch_scale_frame;
    end

    ksf = gaussian_correlation_scale_single(im_patch_scale, im_patch_scale(:,uint8(size(im_patch_scale,2)/2)), p.scale_sigma);
    
    model_hf = ysf'./(ksf+0.1); % lfl: 这个是scale变换的滤波器参数
end