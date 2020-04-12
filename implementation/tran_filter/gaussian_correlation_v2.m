function kf = gaussian_correlation_v2(xf, yf, sigma)
%GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN. They must
%   also be periodic (ie., pre-processed with a cosine window). The result
%   is an MxN map of responses.
%
%   Inputs and output are all in the Fourier domain.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

N = size(xf,1) * size(xf,2);
%     xx_test = bsxfun(@times, xf', xf)
% 错误使用  '
% 未定义 N 维数组的转置。请改用 PERMUTE。
% xx = xf(:)' * xf(:) / N;  %squared norm of x
% yy = yf(:)' * yf(:) / N;  %squared norm of y
% xyf = xf .* conj(yf);
% xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
% xx = arrayfun(@(p) xf(:,:,p)'*xf(:,:,p)/ N, [1:size(xf,3)], 'UniformOutput', false);
% yy = arrayfun(@(p) yf(:,:,p)'*yf(:,:,p)/ N, [1:size(yf,3)], 'UniformOutput', false);
xx  = mtimesx(permute(xf, [2 1 3]), xf) / N;
yy  = mtimesx(permute(yf, [2 1 3]), yf) / N;
xyf = bsxfun(@times, xf, conj(yf));
%cross-correlation term in Fourier domain
% xy = sum(real(ifft2(xyf)), 3);
xy = real(ifft2(xyf));

%calculate gaussian response for all positions, then go back to the
%Fourier domain
kf = fft2(exp(-1 / sigma^2 * (xx + yy - 2 * xy) / N));

end