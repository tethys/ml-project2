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
N = length(labels) - mod(length(labels),50);
X = zeros([N D]);

for i=1:N
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% Iteration
K = 5;

rng(8339);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

layers = [50, 20;
          100, 50;
          200, 50;
          200, 100;
          300, 100;
          400, 200];
      
lambda = 1;
NL = 6;
NN_Pred_scores = zeros( Nk, K, NL );
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
            NN_Pred_scores(:,k, l_index) = PredictNN_complex(Tr, Te, lambda,...
                                                      layers(l_index,:));
        end
    end
    
    %% See prediction performance
    fprintf('Plotting performance..\n');
    
    % and plot all together, and get the performance of each
   % this is to show it in the legend
    results = [];
    for l_index = 1:NL
        results = [results, NN_Pred_scores(:,:,l_index)];
        methodNames{l_index} = sprintf('NN-%d-%d', layers(l_index,1),layers(l_index,2));
    end
    avgTPRList = evaluateMultipleMethods_K_CV( total_Te_y > 0, results, K, true, methodNames );

%% Scale the scores
