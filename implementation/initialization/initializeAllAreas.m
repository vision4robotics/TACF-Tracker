function [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params)
    %% lfl:
    % 用于初始化参数，设置限制的bg_area，之后再限制target_sz，利用target_sz再返回来限制search_area
    % 先利用target_sz计算理论上直接增加padding的backgroud_area(bg_area)大小
    % 比较bg_area比实际图片宽和高，进行限制
    % 把bg_area的面积规范到150^2的大小范围内，并设置response的size
    % 利用规范后的bg_area重新计算target_sz，计算search_area
    %%
	% we want a regular frame surrounding the object
%     im_sz = size(im);
%     if(prod(params.target_sz)/prod(im_sz(1:2)) > 0.05)     % Large target.  From HCF
% 原有方案
%     if prod(params.target_sz) > 50^2
%         params.padding = 2.5;     % Normal target.
%     else
%         params.padding = 4;         
%     end
    if prod(params.target_sz) >= 200^2
        params.padding = 1;     % Normal target.
    elseif (prod(params.target_sz) >= 50^2) && (prod(params.target_sz) < 250^2)
        params.padding = 2.5;
    else
        params.padding = 3; % 原版4 缩小3
    end
    % 原有SWCT[2,4.4], [1.6,3], [1.2, 2.4]
%     if prod(params.target_sz) > 50^2
%         params.padding = 1.6;     % Normal target.
%     else
%         params.padding = 3;         
%     end
    
	avg_dim = sum(params.target_sz) / 2;  % /params.avg_dim_factor; % lfl: /2
	% size from which we extract features
% 	bg_area = round(params.target_sz + avg_dim);                        % lfl: 以目标长宽和的一半扩展padding
	bg_area = round(params.target_sz + avg_dim * params.padding);
    % pick a "safe" region smaller than bbox to avoid mislabeling
	fg_area = round(params.target_sz - avg_dim * params.inner_padding); % lfl: 减少目标(0.2*长宽和的一半)
	% saturate to image size
	if(bg_area(2)>size(im,2)), bg_area(2)=size(im,2)-1; end 
	if(bg_area(1)>size(im,1)), bg_area(1)=size(im,1)-1; end             % lfl: 避免bg_area超出图片大小
	% make sure the differences are a multiple of 2 (makes things easier later in color histograms)
	bg_area = bg_area - mod(bg_area - params.target_sz, 2);             % lfl: 保证bg_area与target_sz的差 = 2的倍数，bg_area可小(-)
	fg_area = fg_area + mod(bg_area - fg_area, 2);                      % lfl: 保证bg_area与fg_area的差 = 2的倍数，fg_area可大(+)

	% Compute the rectangle with (or close to) params.fixedArea and
	% same aspect ratio as the target bbox
	area_resize_factor = sqrt(params.fixed_area/prod(bg_area));         % lfl: 规范bg_area的面积大小，fixed_area = 150^2
	params.norm_bg_area = round(bg_area * area_resize_factor);          % lfl: 计算“规范”（scale变换后的）bg_area大小
	% Correlation Filter (HOG) feature space
	% It smaller that the norm bg area if HOG cell size is > 1
	params.cf_response_size = floor(params.norm_bg_area / params.hog_cell_size); % lfl: 根据norm_bg_area（新的bg_area）设置response的大小，
	% given the norm BG area, which is the corresponding target w and h?
 	norm_target_sz_w = 0.75*params.norm_bg_area(2) - 0.25*params.norm_bg_area(1);
 	norm_target_sz_h = 0.75*params.norm_bg_area(1) - 0.25*params.norm_bg_area(2);% lfl: 在norm_bg_area的约束下重新计算target_sz的大小
%    norm_target_sz_w = params.target_sz(2) * params.norm_bg_area(2) / bg_area(2);
%	norm_target_sz_h = params.target_sz(1) * params.norm_bg_area(1) / bg_area(1);
    params.norm_target_sz = round([norm_target_sz_h norm_target_sz_w]); % lfl: 四舍五入得到在norm_bg_area的约束下的准确目标大小
	% distance (on one side) between target and bg area
	norm_pad = floor((params.norm_bg_area - params.norm_target_sz) / 2);% lfl: 计算规范后padding的大小
	radius = min(norm_pad);
                %     lfl: 示意图
                %     ----------------------------
                %     | bg                       |
                %     |         ----------       |
                %     |         |        |       |       
                %     |<--pad-->| target |       |
                %     |         |        |       |
                %     |         ----------       |
                %     |                          |
                %     ----------------------------
	% norm_delta_area is the number of rectangles that are considered.
	% it is the "sampling space" and the dimension of the final merged resposne
	% it is squared to not privilege any particular direction
	params.norm_delta_area = (2*radius+1) * [1, 1];                                   % lfl: 既然下面还-1，这里其实不用+1的
                                                                                      % 即: 2*radius * [1, 1]
	% Rectangle in which the integral images are computed.
	% Grid of rectangles ( each of size norm_target_sz) has size norm_delta_area.
	params.norm_pwp_search_area = params.norm_target_sz + params.norm_delta_area - 1; % lfl: 再通过规范后的target_sz和pad求搜索域大小

end
