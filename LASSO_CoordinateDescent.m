clc
clear
close all

% User defined parameters
loop_cnt = 10;
tol = 1e-1;

% Load data
data = load('data_std_1.mat');
n = length(data.y);

% Define LAMBDA
lam_max = 2*norm(data.X*(data.y - sum(data.y)/length(data.y)), 'inf');
w_vec = rand(size(data.X, 1), 1);
b = 0;

% Start of Coordinate Descent Algorithm in LASSO
ak = zeros(1, size(data.X, 1));
numnz = zeros(1, loop_cnt);

% Pre compute ak's
for i = 1:length(ak)
    ak(1, i) = 2*data.X(i, :)*data.X(i, :)';
end

% Training loop
for i = 1:loop_cnt
    if i == 1
        lam = lam_max;
    else
        lam = lam_max/(2^(i-1));
    end
    old_obj = 1e16;
    new_obj = objval(data.X, data.y, b, w_vec, lam);
    while old_obj - new_obj > tol
        old_obj = new_obj;
        b = sum(data.y - data.X'*w_vec)/n;
        for k = 1:size(data.X, 1)
            ck = 2*data.X(k,:)*(data.y - data.X'*w_vec - b) + w_vec(k,1)*ak(1,k);
            if ck < -lam
                w_vec(k, 1) = (ck + lam)/ak(1, k);                
            elseif ck > lam
                w_vec(k, 1) = (ck - lam)/ak(1, k);                
            else
                w_vec(k, 1) = 0;                
            end
        end
        new_obj = current_obj(data.X, data.y, b, w_vec, lam);
        del_obj = old_obj - new_obj;
        
        % Stop before diverging
        if del_obj < 0
            break;
        end
    end
    % Store number of non-zero features
    numnz(1, i) = nnz(w_vec);
end

% Plot result
plot(numnz, '-o', 'MarkerFaceColor', 'r');
grid on
xlabel('decreasing \lambda')
ylabel('Number of non-zero features')
legend('number of non-zero features')
title('Evolution of Non Zero Features with decreasing \lambda')

% LASSO objective calculator
function [obj] = current_obj(X, y, b, w, lam)
    E_total = 0;
    R_total = 0;

    for i = 1:size(X, 2)
        E_total = E_total + (w'*X(:, i) + b - y(i, 1))^2;
    end
    
    for i = 1:length(w)
        R_total = R_total + lam * abs(w(i, 1));
    end
    
    obj = E_total + R_total;
end