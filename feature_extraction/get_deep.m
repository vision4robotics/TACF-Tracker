function feature_image = get_deep(im, fparam, gparam, layerInd)
% Extract deep features from VGG

global net
global enableGPU

if isempty(net)
    initial_net();
end

layers = fparam.layers(layerInd);

[im_height, im_width, ~, num_images] = size(im);
temp = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size));
sz_window = size(temp);

feature_image = zeros(size(temp,1), size(temp,2), fparam.nDim(layerInd), num_images);

for k = 1:num_images
    im_temp = single(im(:,:,:,k)); % note: [0, 255] range
    im_temp = imResample(im_temp, net.normalization.imageSize(1:2));
    im_temp = im_temp - net.normalization.averageImage;
    if enableGPU
        im_temp = gpuArray(im_temp);
    end
    % Run the CNN
    res = vl_simplenn(net,im_temp);
    
    if enableGPU
        x = gather(res(layers(1)).x); 
    else
        x = res(layers(1)).x;
    end
    x = imResample(x, sz_window(1:2));
    featuremap = x;
    feature_image(:,:,:,k) = featuremap;
end
end