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

KNN_Pred_5 = zeros( Nk, K );
scores_5 = zeros( Nk, K );
KNN_Pred_9 = zeros( Nk, K );
scores_9 = zeros( Nk, K );
KNN_Pred_15 = zeros( Nk, K );
scores_15 = zeros( Nk, K );
KNN_Pred_19 = zeros( Nk, K );
scores_19 = zeros( Nk, K );
KNN_Pred_25 = zeros( Nk, K );
scores_25 = zeros( Nk, K );
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
    
    %% K-NN classification for 5 neighbours
    fprintf('K-NN classifier for 5 neighbours....\n');
    [KNN_Pred_5(:,k), scores_5(:,k)] = cvKnn(Te.normX', Tr.normX', Tr.y', 19);

    %% K-NN classification for 9 neighbours
    fprintf('K-NN classifier for 9 neighbours....\n');
    [KNN_Pred_9(:,k), scores_9(:,k)] = cvKnn(Te.normX', Tr.normX', Tr.y', 9);

    %% K-NN classification for 15 neighbours
    fprintf('K-NN classifier for 15 neighbours....\n');
    [KNN_Pred_15(:,k), scores_15(:,k)] = cvKnn(Te.normX', Tr.normX', Tr.y', 15);

    %% K-NN classification for 19 neighbours
    fprintf('K-NN classifier for 19 neighbours....\n');
    [KNN_Pred_19(:,k), scores_19(:,k)] = cvKnn(Te.normX', Tr.normX', Tr.y', 19);

    %% K-NN classification for 25 neighbours
    fprintf('K-NN classifier for 25 neighbours....\n');
    [KNN_Pred_25(:,k), scores_25(:,k)] = cvKnn(Te.normX', Tr.normX', Tr.y', 25);
end

%% See prediction performance
fprintf('Plotting performance..\n');

% and plot all together, and get the performance of each
methodNames = {'5-NN', '9-NN', '15-NN', '19-NN', '25-NN'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods_K_CV( total_Te_y > 0, [scores_5,scores_9,scores_15,scores_19,scores_25], K, true, methodNames );
