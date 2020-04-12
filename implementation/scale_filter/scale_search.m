function [target_sz, bg_area,fg_area,area_resize_factor] = scale_search(p,im, pos, base_target_sz, scale_factor, scale_window, scale_model_sz, im_patch_scale, model_hf, min_scale_factor, max_scale_factor, avg_dim_factor, scale_factors)

if p.scale_adaptation
    current_patch = getScalePatch(im, pos, base_target_sz,  scale_factor, scale_window, scale_model_sz, p.hog_scale_cell_size);
    ksf = gaussian_correlation_scale_single(im_patch_scale, current_patch, p.scale_sigma);                    
    scale_response = abs(ifft((model_hf.*ksf)));
    [~, recovered_scale] = max(scale_response(:));

    %set the scale
    scale_factor = scale_factor / scale_factors(recovered_scale);
%                 fprintf('frame %d: recovered scale is %.2f:, current sclae factor is %.2f:\n', frame,recovered_scale,scale_factor)

    if scale_factor < min_scale_factor
        scale_factor = min_scale_factor;
    elseif scale_factor > max_scale_factor
        scale_factor = max_scale_factor;
    end

    % use new scale to update bboxes for target, filter, bg and fg models
    target_sz = round(base_target_sz * scale_factor);
    avg_dim = sum(target_sz)/avg_dim_factor;
    bg_area = round(target_sz + avg_dim);
    if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
    if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end

    bg_area = bg_area - mod(bg_area - target_sz, 2);
    fg_area = round(target_sz - avg_dim * p.inner_padding);
    fg_area = fg_area + mod(bg_area - fg_area, 2);
    % Compute the rectangle with (or close to) params.fixed_area and
    % same aspect ratio as the target bboxgetScaleSubwindow_v1
    area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
end