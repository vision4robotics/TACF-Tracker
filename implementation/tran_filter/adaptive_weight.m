function weight = adaptive_weight(response)

response_temp = response;
max_response = max(response_temp(:));
avg_response = mean(response_temp(:));
std_response = std(response_temp(:));

% weight = (max_response - avg_response)/ std_response;

weight = 1 - exp(- (max_response - avg_response)^2 ./ (2 * std_response) );
end