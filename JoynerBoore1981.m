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
a0(1) = mag(1);
% ijRecord, record j for event i
ijRecord = ones(numRec, 2);
ijRecord(1, 1) = iEvent(1);
for k = 2:length(iEvent)
    if (iEvent(k) == iEvent(k - 1))
        ijRecord(k, 2) = ijRecord(k - 1, 2) + 1;
    else
        tmp = tmp + 1;
    end
    a2magItem(k, tmp) = 1;
    a0(tmp) = mag(k);
    ijRecord(k, 1) = iEvent(k);
end
%% Joyner and Boore, 1981, 
% two-steps method, complete by statistic and machine learning toolbox
% 1st step, determine a, b and h.
h0 = 5.0;
b0 = 0.01;
a0 = a0(:);
x0 = [h0; b0; a0];
logA = log10(accel);
distSquare = dist .* dist;
% model function, modelFun
% logA = a - logR + b1 * logR
% R = sqrt(distSquare + b2 ^ 2)
X = dist;
y = logA;
modelFun = @(b, x) a2magItem * b(3:numEq + 2)  - log10(sqrt(x .* x + b(1) ^ 2)) ...
    + b(2) * sqrt(x .* x + b(1) ^ 2);
mdl = fitnlm(X, y, modelFun, x0);
h = mdl.Coefficients.Estimate(1);
c3 = mdl.Coefficients.Estimate(2);
a = mdl.Coefficients.Estimate(3:end);
sigmaWithin = mdl.RMSE;
%% 2ed step, determin alpha, beta and gamma
y = a;
X = a0;
mdlMag = fitlm(X, y);
c1 = mdlMag.Coefficients.Estimate(1);
c2 = mdlMag.Coefficients.Estimate(2);
sigmaBetween = mdlMag.RMSE;
%% log(A) = c1 + c2 * Magnitude  - log(R) + c3 * R
% R = sqrt(dist .* dist + h ^ 2)
sigmaSum = sqrt(sigmaBetween ^ 2 + sigmaWithin ^ 2);

%% conclusion
% same coefficients and sigma as Joyner and Boore, 1981
% congratulations!
%% end time
toc;