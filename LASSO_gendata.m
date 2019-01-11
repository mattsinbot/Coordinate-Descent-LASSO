clc
clear
close all

% Coordinate Descent Algorihthm Implementation (Inefficient)
N = 250; % number of data points
d = 80;  % dim of features
k = 10;  % dim of non-zero features
sig = 1; % sigma for generating moise

% Generate synthetic training data
b = 0;
X = normrnd(0, sig, d, N);
w = zeros(d, 1);
for i = 1:k
    if mod(i, 2) == 0
        w(i, 1) = 10;
    else
        w(i, 1) = -10;
    end
end
noise = normrnd(0, 1, N, 1);
y = X'*w + b + noise;

% Save variable X and y
save('data_sig1.mat', 'X', 'y', 'w');