function [ysf, scale_factors, scale_model_sz, min_scale_factor, max_scale_factor, base_target_sz, scale_window, scale_factor] = scale_adaptation_init(im, p, target_sz, bg_area)
if p.scale_adaptation
    % Code from DSST
    scale_factor = 1;
    base_target_sz = target_sz;
    scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
    ss = (1:p.num_scales) - ceil(p.num_scales/2);

    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);

    ys=zeros(size(ys));
    ys(1)=1;
    ysf = ((fft(ys)));
    if mod(p.num_scales,2) == 0
        scale_window = single(hann(p.num_scales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(p.num_scales));
    end

    ss = 1:p.num_scales;
    scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss); % lfl: 生成一系列尺度因子

    % lfl: scale_model_max_area = 32*16
    % norm_target_sz与scale_model_max_area比较
    if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
        p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
    end
    % lfl: 利用scale_model_max_area来规范大小，计算出当前norm_target_sz对应的scale_model_sz
    scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
    % find maximum and minimum scales
    min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
    max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));
end