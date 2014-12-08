close all
clear all
clc

%% Load both features and training images
load train_feats;
load train_imgs;

%% Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% Iteration
K = 5;

rng(8339);
N = size(labels,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

LR_Pred = zeros( Nk, K );
total_Te_y = [];

for k = 1 : K
    %% Get k'th subgroup in test, the rest in train
    Te.idxs = idxCV(k,:);
    Tr.idxs = idxCV([1:k-1 k+1:end],:);
    Tr.idxs = Tr.idxs(:);

    Tr.X = X(Tr.idxs,:);
    Tr.y = labels(Tr.idxs);

    Te.X = X(Te.idxs,:);
    Te.y = labels(Te.idxs);
    total_Te_y = [total_Te_y Te.y];
    
    %% Normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X);
    Te.normX = normalize(Te.X, mu, sigma);
    
    %% Logistic Regression classification
    fprintf('Training Logistic Regression classifier....\n');
    beta = logisticRegression(Tr.y, Tr.normX, 1e-2);
    fprintf('Predicting for Logistic Regression classifier....\n');
    pHatn = sigmoid(Te.normX*beta);
    LR_Pred(:,k) = 2*pHatn - 1;

end

%% See prediction performance
fprintf('Plotting performance..\n');

% and plot all together, and get the performance of each
methodNames = {'Logistic Regression'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods_K_CV( total_Te_y > 0, LR_Pred, K, true, methodNames );
