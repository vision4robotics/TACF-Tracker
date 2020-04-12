function feature_image = get_saliency( im, fparam, gparam )

[im_height, im_width, ~, num_images] = size(im);
% [im_height, im_width, num_im_chan, num_images] = size(im);
temp = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size), fparam.nDim, num_images, 'single');
feature_image = zeros(size(temp,1), size(temp,2), fparam.nDim, num_images);

for k = 1:num_images
    rgb=imresize(im(:,:,:,k),gparam.factor);
    myFFT = fft2(rgb);
    myLogAmplitude = log(abs(myFFT));
    myPhase = angle(myFFT);
    mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
    saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
    saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
    im_saliency = saliencyMap;
    feature_image(:,:,:,k) = mexResize(im_saliency, [size(temp,1) size(temp,2)],'auto');
end