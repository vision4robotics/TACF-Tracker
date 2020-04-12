function [local_max] = localmax_nonmaxsup2d(response)
% find local 1st and 2nd maximum in 2d matrix
% code borrow from CSRDCF

BW = imregionalmax(response);
CC = bwconncomp(BW);

local_max = [max(response(:)) 0];
if length(CC.PixelIdxList) > 1
    local_max = zeros(length(CC.PixelIdxList));
    for i = 1:length(CC.PixelIdxList)
        local_max(i) = response(CC.PixelIdxList{i}(1));
    end
    local_max = sort(local_max, 'descend');
end

end  % endfunction