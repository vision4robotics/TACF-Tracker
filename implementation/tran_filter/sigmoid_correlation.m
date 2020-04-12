function kf = sigmoid_correlation(xf, yf, sigma, c)
%SIGMOID_CORRELATION Polynomial Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a polynomial kernel with constant A and exponent B, for all
%   relative shifts between input images XF and YF, which must both be MxN.
%   They must also be periodic (ie., pre-processed with a cosine window).
%   The result is an MxN map of responses.
%
%   Inputs and output are all in the Fourier domain.
	
	%cross-correlation term in Fourier domain
	xyf = xf .* conj(yf);
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
	
	%calculate polynomial response for all positions, then go back to the
	%Fourier domain
	kf = fft2(tanh(sigma * xy / numel(xf) + c));

end

