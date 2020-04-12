function PMER = calPMER(response)

% Calculates the PMER (Peak-Median Energy Ratio) of the response map
% in order to obtain a measure of peak strength 

% load('sample_data/correlation_mat');

% Get location of the max and min peak
[xmax, ymax] = ind2sub(size(response),find(response == max(response(:)), 1));

% Get max and min peak value
res_max = response(xmax,ymax);
res_med = median(response(:));

% Peak sharpness of the response map
num = (res_max - res_med)^2;

% Overall fluctuations of the response map
fluc_map = (response - res_med).^ 2;
den = mean(fluc_map(:)); % version1
% den = std(fluc_map(:)); % version2

PMER = num / den;
end
