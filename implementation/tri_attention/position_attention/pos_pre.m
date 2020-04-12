function pre_output = pos_pre(R, win)

% dimension axis sum
R_win = bsxfun(@times, R, win);
pos_sum = sum(R_win,3);
map_norm = normalize_img(pos_sum);
map_mean = map_norm - mean(map_norm(:));
map = max(map_mean,0);
pre_output = exp(map)-0.5;

end

function out = normalize_img(img)
% normalize the 2d matrix
% code borrow from CSRDCF

min_val = min(img(:));
max_val = max(img(:));

if (max_val - min_val) > 0
    out = (img - min_val)/(max_val - min_val);
else
    out = zeros(size(img));
end

end  % endfunction