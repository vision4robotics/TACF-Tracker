function setup_paths()

% Add the neccesary paths
[pathstr,~,~] = fileparts(mfilename('fullpath'));
addpath(genpath([pathstr '/utils/']));
addpath(genpath([pathstr '/model/']));
addpath(genpath([pathstr '/implementation/']));
addpath(genpath([pathstr '/feature_extraction/']));
addpath(genpath([pathstr '/external_libs/']));
addpath(genpath([pathstr '/UAV123_utils/']));

end