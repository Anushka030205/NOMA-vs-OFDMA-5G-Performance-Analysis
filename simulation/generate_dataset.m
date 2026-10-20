clc; clear;

N = 10000;

SNR_dB = randi([0 30], N, 1);
labels = zeros(N,1);

for i = 1:N
    
    snr = 10^(SNR_dB(i)/10);
    
    h1 = (randn + 1j*randn)/sqrt(2);
    h2 = (randn + 1j*randn)/sqrt(2);
    
    g1 = abs(h1)^2;
    g2 = abs(h2)^2;
    
    if g1 > g2
        temp = g1; g1 = g2; g2 = temp;
    end
    
    a1 = 0.8; a2 = 0.2;
    
    noma = log2(1 + (a1*snr*g1)/(a2*snr*g1 + 1)) + log2(1 + a2*snr*g2);
    ofdm = log2(1 + snr*g1) + log2(1 + snr*g2);
    
    labels(i) = noma > ofdm;
    
end

data = [SNR_dB labels];

writematrix(data, 'dataset.csv');

disp('Dataset created!');