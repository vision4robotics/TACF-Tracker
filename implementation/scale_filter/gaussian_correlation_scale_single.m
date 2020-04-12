function kf = gaussian_correlation_scale_single(base, y, sigma)
	
	k = zeros(1,size(base,2));
    N = numel(y);
    for i =1:size(base,2)
        n = sum(sum(sum((base(:,i) - y).^2)))/N^2;
        k(i) = exp(-n / sigma^2 ) ;
    end
    
    kf = fft(k');
end
