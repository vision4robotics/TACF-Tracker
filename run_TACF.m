function results = run_TACF(seq,res_path, bSaveImage)

setup_paths;
params.visualization = 1;

%% New add-on
params.position_attention_avail = 1;
params.pos_lr = 0.02;
params.channel_attention_avail = 1;
params.chann_wlr = 0.022;
params.dim_act_threshold = 0.3;
% params.temporal_attention_avail = 1;
params.two_feature = 1;
% contextual attention
params.context_att = 1;
% learning intervals for context similarity /frame
params.bgl_interv = 4;

params.hog_cell_size = 4;
params.fixed_area = 200^2;   % 150^2           % standard area to which we resize the target
params.n_bins = 2^5;                           % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.02;               % bg and fg color models learning rate
params.lambda_scale = 0.1;                     % regularization weight
params.scale_sigma_factor = 1/16;
params.scale_sigma = 0.1;
params.merge_factor = 0.3;

% fixed setup
params.hog_scale_cell_size = 4;                % Default DSST=4
params.scale_model_factor = 1.0;

params.feature_type = 'fhog';
params.scale_adaptation = true;
params.grayscale_sequence = false;	          % suppose that sequence is colour
params.merge_method = 'const_factor';

params.img_files = seq.s_frames;
params.img_path = '';

s_frames = seq.s_frames;
params.s_frames = s_frames;
params.video_path = seq.video_path;
im = imread([s_frames{1}]);
% grayscale sequence? --> use 1D instead of 3D histograms
if(size(im,3)==1)
    params.grayscale_sequence = true;
end

region = seq.init_rect;

if(numel(region)==8)
    % polygon format (VOT14, VOT15)
    [cx, cy, w, h] = getAxisAlignedBB(region);
else % rectangle format (WuCVPR13)
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x+w/2;
    cy = y+h/2;
end

% init_pos is the centre of the initial bounding box
params.init_pos = [cy cx];
params.target_sz = round([h w]);
% defines inner area used to sample colors from the foreground
params.inner_padding = 0.2;

[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

% HOG feature parameters
hog_params.nDim  = 31;
% CN feature parameters
cn_params.nDim = 11;
% Gray feature parameters
gray_params.nDim = 1;
% Saliency feature parameters
saliency_params.nDim = 3;
% Deep feature parameters
params.indLayers = [37, 28, 19];%   The CNN layers Conv3-4 in VGG Net
deep_params.nDim = [512, 512, 256];
deep_params.layers = params.indLayers;
%   handcrafted parameters
Feat1 = 'fhog'; % fhog, cn, gray, saliency, handcrafted_assem fhog_cn
switch Feat1
    case 'conv3'
        params.layerInd{1} = 3;
        params.feat1dim = deep_params.nDim(1);
    case 'conv4'
        params.layerInd{1} = 2;
        params.feat1dim = deep_params.nDim(2);
    case 'conv5'
        params.layerInd{1} = 1;
        params.feat1dim = deep_params.nDim(3);
    case 'fhog'
        params.layerInd{1} = 0;
        params.feat1dim = hog_params.nDim;
    case 'cn'
        params.layerInd{1} = 0;
        params.feat1dim = cn_params.nDim;
    case 'handcrafted_assem'
        params.layerInd{1} = 0;
        params.feat1dim = hog_params.nDim + cn_params.nDim;
end

if params.two_feature
    Feat2 = 'cn'; % fhog, cn, gray, saliency, handcrafted_assem
    switch Feat2
        case 'conv3'
            params.layerInd{2} = 3;
            params.feat2dim = deep_params.nDim(1);
        case 'conv4'
            params.layerInd{2} = 2;
            params.feat2dim = deep_params.nDim(2);
        case 'conv5'
            params.layerInd{2} = 1;
            params.feat2dim = deep_params.nDim(3);
        case 'fhog'
            params.layerInd{2} = 0;
            params.feat2dim = hog_params.nDim;
        case 'cn'
            params.layerInd{2} = 0;
            params.feat2dim = cn_params.nDim;
        case 'handcrafted_assem'
            params.layerInd{2} = 0;
            params.feat2dim = hog_params.nDim + cn_params.nDim;
    end
    params.feat_type = {Feat1, Feat2};
else
    params.feat_type = {Feat1};
end

params.t_global.type_assem = 'fhog_cn'; % fhog_cn, fhog_gray,fhog_cn_gray_saliency, fhog_gray_saliency,fhog_cn_gray,fhog_gray
switch params.t_global.type_assem
    case 'fhog_cn_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_cn_gray'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim;
    case 'fhog_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_gray'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim;
    case 'fhog_cn'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim;
end

params.t_features = {struct('getFeature_fhog',@get_fhog,...
    'getFeature_cn',@get_cn,...
    'getFeature_gray',@get_gray,...
    'getFeature_saliency',@get_saliency,...
    'getFeature_deep',@get_deep,...
    'getFeature_handcrafted',@get_handcrafted,...
    'hog_params',hog_params,...
    'cn_params',cn_params,...
    'gray_params',gray_params,...
    'saliency_params',saliency_params,...
    'deep_params',deep_params,...
    'handcrafted_params',handcrafted_params)};

params.t_global.w2c_mat = load('w2c.mat');
params.t_global.factor = 0.2; % for saliency
params.t_global.cell_size = 4;
params.t_global.cell_selection_thresh = 0.75^2;

params.lambda1 = 1e-4;    %
params.lambda2 = 1/(16^2); %

kernel_type{1} = 'gaussian';
kernel_type{2} = 'polynomial';
params.kernel_type = kernel_type;
params.output_sigma_factor = {1/40, 1/16};
params.tran_sigma = {0.5, 0.5};
params.polya = {1,1};
params.polyb = {7,2};

params.learning_rate_cf = 0.01;

params.num_scales = 33;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
params.learning_rate_scale = 0.004;

% start the tracking
results = tracker(params, im, bg_area, fg_area, area_resize_factor);

end