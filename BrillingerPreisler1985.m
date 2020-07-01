% script file: jb1981.m
% test the method of Joyner and Boore, 1981

% clear workspace
clear;clc;
dbstop if error
%% initial time
tic;

%% load data, table
% delete records only recorded on one station
load JoynerBoore1981.mat
% expand to double vectors
mag = JoynerBoore1981.mag; % magnitude, unit: Mw
dist = JoynerBoore1981.dist; % distance, unit: km
accel = JoynerBoore1981.accel; % acceleration, unit: g
% serial number of events, iEvent
iEvent = JoynerBoore1981.event;
% numEq, number of earthquakes; numRec, number of records
numRec = length(accel);
numEq = length(unique(iEvent));
% a2magItem, from a to magItem in fun,
% a, refer to Joyner and Boore, 1981, eq. (2)
% a2magItem, transform from magnitude to magnitude items
a2magItem = zeros(numRec, numEq);
% magnitudes of events
a0 = zeros(numEq, 1);
tmp = 1;
a2magItem(1, tmp) = 1;
a0(1) = 1; % category of magnitude, from 1 to end
% ijRecord, record j for event i
ijRecord = ones(numRec, 2);
ijRecord(1, 1) = iEvent(1);
for k = 2:length(iEvent)
    if (iEvent(k) == iEvent(k - 1))
        ijRecord(k, 2) = ijRecord(k - 1, 2) + 1;
    else
        a0(tmp + 1) = a0(tmp) + 1;
        tmp = tmp + 1;
    end
    a2magItem(k, tmp) = 1;
    
    ijRecord(k, 1) = iEvent(k);
end
%% Brillinger and Preisler, 1985
% non-linear mixed model regression by nlmefit function
% beta = nlmefit(X, y, group, V, modelfun, beta0);
logA = log10(accel);
y = logA;
distSquare = dist .* dist;
X = [mag, distSquare];
magCate = a2magItem * a0;
group = magCate;
beta0 = [-1.02, 0.25, -0.0025, 7.3];
V = [];
modelFun = @(phi, x) phi(1) + phi(2) * x(:, 1) - ...
    0.5 * log10(x(:, 2) + phi(4) ^ 2) + ...
    phi(3) * sqrt(x(:, 2) + phi(4) ^2);
opt = statset('display', 'iter');
% model function, modelFun
% logA = a - logR + b1 * logR
% R = sqrt(distSquare + b2 ^ 2)
[beta, psi, stats, B] = nlmefit(X, y, group, V, modelFun, beta0, ...
    'REparamsSelect', 1, 'errorModel', 'constant', ...
    'ApproximationType', 'RELME', ...
    'options', opt);
sigmaWithin = stats.errorparam;
sigmaBetween = sqrt(psi);
sigmaSum = sqrt(sigmaWithin ^ 2 + sigmaBetween ^ 2);
%% plot figure one in this paper,
% predicted ground motion of earthquake whose magnitude is Mw 7.0
fitted_model = @(x) beta(1) + beta(2) * x(:, 1) - ...
    0.5 * log10(x(:, 2) + beta(4) ^ 2) + ...
    beta(3) * sqrt(x(:, 2) + beta(4) ^ 2);
distPlot = linspace(1, 100, 100)';
magPlot = 7.0 * ones(size(distPlot));
x = [magPlot, distPlot .* distPlot];
adjust = icdf('normal', 0.025, 0, 1) * sigmaSum;

figure(1);
loglog(distPlot, 10 .^ fitted_model(x), 'k-', ...
    'linewidth', 2);
hold on; grid on;
loglog(distPlot, 10 .^ (fitted_model(x) + sigmaSum * adjust), 'g-', ...
    'linewidth', 1);
loglog(distPlot, 10 .^ (fitted_model(x) - sigmaSum * adjust), 'r-', ...
    'linewidth', 1);
% configuration 
axis([1, 500, 0.005, 5]);
xticks([1, 5, 10, 50, 100, 500]);
yticks([0.005, 0.05, 0.5, 1, 5]);
xlabel('Distance(km)'); ylabel('PGA(g)');
%% conclusion
% log10(A_ij) = -0.9466(0.2250) + 0.236(0.038) * Magnitude - log10(sqrt(dist ^ 2 + h ^ 2))
% ...           -0.0020(0.0004) * sqrt(dist ^ 2 + h ^ 2) + 
% ...           + 0.0692 * z_i + 0.22 * z_ij
% h = 7.10(1.28)
% beta = [  -0.9466    0.2359   -0.0020    7.1014 ]';
% PSI, estimated covariance of random effects
% standard error of beta = [0.2250    0.0381    0.0004    1.2778];
% stats.sebeta, standard error of beta
% stats.rmse, stats.errorparam, sigma of z_ij
% stats.covb, covariance of beta
% B, one realization of random effects

%% Brillinger and Preisler, 1985
% regression parameters and confidence interval
% logA_ij = -1.229(0.196) + 0.277(0.034) * M_i - log(r_ij) 
%           - 0.00231(0.00062)r_ij + 0.1223(0.0305)* Z_i 
%           + 0.2284(0.0127) * Z_ij
% r_ij ^ 2 = d_ij ^ 2 + 6.65(2.612) ^ 2
betaBP = [-1.229, 0.277, -0.00231, 6.65];
sigmaSumBP = sqrt(0.1223 ^ 2 + 0.2287 ^ 2);
modelBP = @(x) betaBP(1) + betaBP(2) * x(:, 1) - ...
    0.5 * log10(x(:, 2) + betaBP(4) ^ 2) + ...
    betaBP(3) * sqrt(x(:, 2) + betaBP(4) ^ 2);
loglog(distPlot, 10 .^ modelBP(x), 'k--', ...
    'linewidth', 2);
hold on; grid on;
loglog(distPlot, 10 .^ (modelBP(x) + sigmaSumBP * adjust), 'g--', ...
    'linewidth', 1);
loglog(distPlot, 10 .^ (modelBP(x) - sigmaSumBP * adjust), 'r--', ...
    'linewidth', 1);
legend({'median of MATLAB ML model', ...
    '2.5% percentile of MATLAB ML model', ...
    '97.5% percentile of MATLAB ML model', ...
    'median of model BP', ...
    '2.5% percentile of BP', ...
    '97.5% percentile of BP'}, 'location', 'sw');

%% save picture
print('BrillingerPresiler1985', '-dtiff')
%% end time
hold off;
toc;