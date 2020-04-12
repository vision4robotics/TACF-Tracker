function TACF_Demo(~)
close all;
clear;
clc;

%% **Need to change**
type_of_assessment = 'UAV123_10fps';
tracker_name = 'TACF';
%% Load video information
seq = load_video_information(type_of_assessment);

% main function
result = run_TACF(seq);

% save results
results = cell(1,1);
results{1} = result;
results{1}.len = seq.len;
results{1}.startFrame = seq.st_frame;
results{1}.annoBegin = seq.st_frame;

% save results to specified folder
save_dir = '.\Test_one_seq\';
save_res_dir = [save_dir, tracker_name, '_results\'];
save_pic_dir = [save_res_dir, 'res_picture\'];
if ~exist(save_res_dir, 'dir')
    mkdir(save_res_dir);
    mkdir(save_pic_dir);
end
save([save_res_dir, seq.video_name, '_', tracker_name], 'results');

% plot precision figure
show_visualization = 1;
precision_plot_save(results{1}.res, seq.ground_truth, seq.video_name, save_pic_dir, show_visualization);