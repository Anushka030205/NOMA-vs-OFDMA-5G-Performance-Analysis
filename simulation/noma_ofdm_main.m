clc; clear; close all;

%% Parameters
numUsers = 2;
N = 1e4;
snr_dB = 0:2:20;
snr = 10.^(snr_dB/10);
Pt = 1;

a1 = 0.75; % far user
a2 = 0.25; % near user

% Initialize
SE_NOMA = zeros(size(snr));
SE_OFDMA = zeros(size(snr));

BER_NOMA = zeros(size(snr));
BER_OFDMA = zeros(size(snr));

fairness_noma = zeros(size(snr));
fairness_ofdma = zeros(size(snr));

throughput_noma = zeros(size(snr));
throughput_ofdma = zeros(size(snr));

%% ================= MAIN SIMULATION =================
for i = 1:length(snr)

    noiseVar = Pt/snr(i);

    %% Data
    d1 = randi([0 1],N,1);
    d2 = randi([0 1],N,1);

    s1 = 2*d1 - 1;
    s2 = 2*d2 - 1;

    %% Channel
    h1 = 0.3*(randn(N,1)+1j*randn(N,1))/sqrt(2); % weak user
    h2 = 1.5*(randn(N,1)+1j*randn(N,1))/sqrt(2); % strong user

    %% Noise
    n1 = sqrt(noiseVar/2)*(randn(N,1)+1j*randn(N,1));
    n2 = sqrt(noiseVar/2)*(randn(N,1)+1j*randn(N,1));

    %% -------- NOMA --------
    x = sqrt(a1*Pt).*s1 + sqrt(a2*Pt).*s2;

    y1 = h1 .* x + n1;
    y2 = h2 .* x + n2;

    % Detection
    y1_eq = y1 ./ h1;
    d1_hat = real(y1_eq) > 0;

    y2_eq = y2 ./ h2;
    d1_est_near = real(y2_eq) > 0;
    s1_est = 2*d1_est_near - 1;

    y2_sic = y2_eq - sqrt(a1*Pt).*s1_est;
    d2_hat = real(y2_sic) > 0;

    % BER
    BER_NOMA(i) = (sum(d1 ~= d1_hat) + sum(d2 ~= d2_hat)) / (2*N);

    % Rate (Sum Rate / Spectral Efficiency)
    SINR1 = (a1*Pt*abs(h1).^2) ./ (a2*Pt*abs(h1).^2 + noiseVar);
    SINR2 = (a2*Pt*abs(h2).^2) ./ noiseVar;

    R1 = mean(log2(1 + SINR1));
    R2 = mean(log2(1 + SINR2));

    SE_NOMA(i) = R1 + R2;

    % Fairness
    fairness_noma(i) = (R1 + R2)^2 / (2*(R1^2 + R2^2));

    % Throughput (with PER)
    per_noma = 0.05 + 0.05*rand();
    throughput_noma(i) = SE_NOMA(i)*(1 - per_noma);

    %% -------- OFDMA --------
    P_user = Pt / numUsers;

    y1_ofdm = h1 .* sqrt(P_user) .* s1 + n1;
    y2_ofdm = h2 .* sqrt(P_user) .* s2 + n2;

    d1_hat_ofdm = real(y1_ofdm ./ h1) > 0;
    d2_hat_ofdm = real(y2_ofdm ./ h2) > 0;

    BER_OFDMA(i) = (sum(d1 ~= d1_hat_ofdm) + sum(d2 ~= d2_hat_ofdm)) / (2*N);

    SNR1 = (P_user * abs(h1).^2) ./ noiseVar;
    SNR2 = (P_user * abs(h2).^2) ./ noiseVar;

    R1_ofdm = mean(log2(1 + SNR1));
    R2_ofdm = mean(log2(1 + SNR2));

    SE_OFDMA(i) = (R1_ofdm + R2_ofdm) / numUsers;

    fairness_ofdma(i) = (R1_ofdm + R2_ofdm)^2 / ...
                        (2*(R1_ofdm^2 + R2_ofdm^2));

    per_ofdma = 0.1 + 0.05*rand();
    throughput_ofdma(i) = SE_OFDMA(i)*(1 - per_ofdma);

end

%% ================= ML DATASET =================
numSamples = 5000;

features = zeros(numSamples,4);
labels = zeros(numSamples,1);

for j = 1:numSamples
    
    snr_dB_rand = randi([0 20]);
    snr_rand = 10^(snr_dB_rand/10);
    noiseVar = Pt/snr_rand;

    h1 = 0.3*(randn + 1j*randn)/sqrt(2);
    h2 = 1.5*(randn + 1j*randn)/sqrt(2);

    g1 = abs(h1)^2;
    g2 = abs(h2)^2;

    if g1 > g2
        temp = g1; g1 = g2; g2 = temp;
    end

    a1_rand = 0.7 + 0.25*rand();

    if rand < 0.2
        a1_rand = 0.5; % allow OFDMA cases
    end

    a2_rand = 1 - a1_rand;

    % NOMA
    SINR1 = (a1_rand*Pt*g1)/(a2_rand*Pt*g1 + noiseVar);
    SINR2 = (a2_rand*Pt*g2)/(noiseVar);
    noma = log2(1 + SINR1) + log2(1 + SINR2);

    % OFDMA
    P_user = Pt/2;
    ofdm = (log2(1 + P_user*g1/noiseVar) + ...
            log2(1 + P_user*g2/noiseVar))/2;

    features(j,:) = [snr_dB_rand g1 g2 a1_rand];
    labels(j) = noma > ofdm;

end

dataset = [features labels];
writematrix(dataset,'dataset.csv');

disp('Dataset created for ML!');

%% ================= PLOTS =================

% Sum Rate
figure;
plot(snr_dB, SE_NOMA,'r-o','LineWidth',2); hold on;
plot(snr_dB, SE_OFDMA,'b-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('bits/s/Hz');
title('Sum Rate Comparison');
legend('NOMA','OFDMA');

% BER
figure;
semilogy(snr_dB, BER_NOMA,'r-o','LineWidth',2); hold on;
semilogy(snr_dB, BER_OFDMA,'b-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title('BER Comparison');
legend('NOMA','OFDMA');

% Fairness
figure;
plot(snr_dB, fairness_noma,'r-o','LineWidth',2); hold on;
plot(snr_dB, fairness_ofdma,'b-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('Jain Fairness Index');
title('Fairness Comparison');
legend('NOMA','OFDMA');

% Throughput
figure;
plot(snr_dB, throughput_noma,'r-o','LineWidth',2); hold on;
plot(snr_dB, throughput_ofdma,'b-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('Throughput');
title('Throughput Comparison');
legend('NOMA','OFDMA');