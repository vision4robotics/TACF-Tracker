function map = pos_attention(R, win, frame, target_sz, trans_vec)
pre_output = pos_pre(R, win);
size_R = size(R);
if frame < 3
    map = pre_output;
else
    if norm(trans_vec) == 0
        map = pre_output;
    else
        % 最大值定位0.5
        % 4太差了，3稍微上去了但是没用
        motion_level = norm(trans_vec) / norm(target_sz)* 2; 
%         motion_level = norm(trans_vec) / norm(target_sz) * 2;
        % 求取单位向量，the motion_direct should resize accorinding to the traget_sz
        resize_vec = trans_vec ./ target_sz;
        motion_direct = resize_vec/norm(resize_vec);
        % 移动向量取整
        motion_vec = round(motion_level * size_R(1:2) .* motion_direct);
        motion_map = circshift(pre_output, motion_vec);
        map = pre_output + motion_level * motion_map;
    end
end
end