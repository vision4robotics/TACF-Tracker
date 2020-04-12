function feature_image = get_handcrafted( im, fparam, gparam )

if ~isfield(fparam, 'nOrients')
    fparam.nOrients = 9;
end

[im_height, im_width, ~, num_images] = size(im);

temp = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size)); 
feature_image = zeros(size(temp,1), size(temp,2), fparam.nDim, num_images);

switch gparam.type_assem
    case 'fhog_cn_gray_saliency'
        for k = 1:num_images
            im_temp = mexResize(im(:,:,:,k), [size(temp,1) size(temp,2)],'auto'); 

            hog_image = double(fhog(single(im(:,:,:,k)) / 255, gparam.cell_size, fparam.nOrients));
            hog_image(:,:,end) = []; %the last dimension is all 0 so we can discard it

            gray_image = rgb2gray(im_temp);
            gray_image= double(gray_image)/255;
            gray_image = gray_image - mean(gray_image(:));

            cn_image = im2c(double(im_temp(:,:,:)),gparam.w2c_mat.w2c,-2);

            rgb = mexResize(im(:,:,:,k),gparam.factor,'auto');
            myFFT = fft2(rgb);
            myLogAmplitude = log(abs(myFFT));
            myPhase = angle(myFFT);
            mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
            saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
            saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
            saliency_image = saliencyMap;
            saliency_image = mexResize(saliency_image, [size(temp,1) size(temp,2)],'auto');

            feat_temp1 = cat(3, hog_image, gray_image);
            feat_temp2 = cat(3, cn_image, saliency_image);

            feature_image(:,:,:,k) = cat(3, feat_temp1, feat_temp2);
        end
    case 'fhog_gray_saliency'
        for k = 1:num_images
            im_temp = mexResize(im(:,:,:,k), [size(temp,1) size(temp,2)],'auto'); 

            hog_image = double(fhog(single(im(:,:,:,k)) / 255, gparam.cell_size, fparam.nOrients));
            hog_image(:,:,end) = []; %the last dimension is all 0 so we can discard it

            gray_image = rgb2gray(im_temp);
            gray_image= double(gray_image)/255;
            gray_image = gray_image - mean(gray_image(:));

            rgb = mexResize(im(:,:,:,k),gparam.factor,'auto');
            myFFT = fft2(rgb);
            myLogAmplitude = log(abs(myFFT));
            myPhase = angle(myFFT);
            mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
            saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
            saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
            saliency_image = saliencyMap;
            saliency_image = mexResize(saliency_image, [size(temp,1) size(temp,2)],'auto');

            feat_temp1 = cat(3, hog_image, gray_image);

            feature_image(:,:,:,k) = cat(3, feat_temp1, saliency_image);
        end
    case 'fhog_cn_gray'
        for k = 1:num_images
            im_temp = mexResize(im(:,:,:,k), [size(temp,1) size(temp,2)],'auto'); 

            hog_image = double(fhog(single(im(:,:,:,k)) / 255, gparam.cell_size, fparam.nOrients));
            hog_image(:,:,end) = []; %the last dimension is all 0 so we can discard it

            gray_image = rgb2gray(im_temp);
            gray_image= double(gray_image)/255;
            gray_image = gray_image - mean(gray_image(:));

            cn_image = im2c(double(im_temp(:,:,:)),gparam.w2c_mat.w2c,-2);
            
            feat_temp1 = cat(3, hog_image, gray_image);

            feature_image(:,:,:,k) = cat(3, feat_temp1, cn_image);
        end
    case 'fhog_gray'
        for k = 1:num_images
            im_temp = mexResize(im(:,:,:,k), [size(temp,1) size(temp,2)],'auto'); 

            hog_image = double(fhog(single(im(:,:,:,k)) / 255, gparam.cell_size, fparam.nOrients));
            hog_image(:,:,end) = []; %the last dimension is all 0 so we can discard it

            gray_image = rgb2gray(im_temp);
            gray_image= double(gray_image)/255;
            gray_image = gray_image - mean(gray_image(:));

            feature_image(:,:,:,k) = cat(3, hog_image, gray_image);
        end
    case 'fhog_cn'
        for k = 1:num_images
            im_temp = mexResize(im(:,:,:,k), [size(temp,1) size(temp,2)],'auto'); 

            hog_image = double(fhog(single(im(:,:,:,k)) / 255, gparam.cell_size, fparam.nOrients));
            hog_image(:,:,end) = []; %the last dimension is all 0 so we can discard it

            cn_image = im2c(double(im_temp(:,:,:)),gparam.w2c_mat.w2c,-2);

            feature_image(:,:,:,k) = cat(3, hog_image, cn_image);
        end
end
        

