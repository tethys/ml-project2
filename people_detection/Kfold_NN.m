close all
clear all
clc

%% Load both features and training images
load train_feats;
load train_imgs;

clear imgs;
%% Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
N = length(labels) - mod(length(labels),100);
X = zeros([N D]);

for i=1:N
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% Iteration
K = 10;

rng(8339);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

lambda = [0.01,0.1,1,10];
NL = 4;
NN_Pred_scores = zeros( Nk, K,NL );
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
        
        for l_index = 1:NL
            NN_Pred_scores(:,k, l_index) = PredictNN(Tr, Te, lambda(l_index));
        end
    end
    
    %% See prediction performance
    fprintf('Plotting performance..\n');
    
    % and plot all together, and get the performance of each
   % this is to show it in the legend
    results = [];
    for l_index = 1:NL
        results = [results, NN_Pred_scores(:,:,l_index)];
        methodNames{l_index} = sprintf('NN-alpha-%1.1f', lambda(l_index));
    end
    avgTPRList = evaluateMultipleMethods_K_CV( total_Te_y > 0, results, K, true, methodNames );

%% Scale the scores
