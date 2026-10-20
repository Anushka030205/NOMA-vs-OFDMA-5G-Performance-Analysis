clc; clear; close all;

N = 10000; % number of samples
SNR_dB = randi([0 30], N, 1);
users = randi([2 5], N, 1);

noma_rate = zeros(N,1);
ofdm_rate = zeros(N,1);

for i = 1:N
    snr = 10^(SNR_dB(i)/10);
    u = users(i);

    % OFDM Sum Rate
    ofdm_rate(i) = u * log2(1 + snr);

    % NOMA Sum Rate (simplified)
    noma_rate(i) = log2(1 + snr*u);
end

% Features and Labels
X = [SNR_dB users];
Y = noma_rate - ofdm_rate; % difference (target)

data = [X Y];

writematrix(data, 'noma_ofdm_dataset.csv');

disp('Dataset generated!');