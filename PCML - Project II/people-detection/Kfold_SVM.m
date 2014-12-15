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

%rng(8339);
N = size(labels,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

Lin_SVM_Pred = zeros( Nk, K );
Quad_SVM_Pred = zeros( Nk, K );
Cub_SVM_Pred = zeros( Nk, K );
RBF_SVM_Pred = zeros( Nk, K );
Sigm_SVM_Pred = zeros( Nk, K );
Lin_SVM_scores = zeros( Nk, K );
Quad_SVM_scores = zeros( Nk, K );
Cub_SVM_scores = zeros( Nk, K );
RBF_SVM_scores = zeros( Nk, K );
Sigm_SVM_scores = zeros( Nk, K );
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
    
    %% Linear kernel SVM classification
    fprintf('Training linear kernel SVM....\n');
    Lin_cls = svmtrain(Tr.y, Tr.normX, '-t 0');
    fprintf('Predicting for linear kernel SVM....\n');
    [Lin_SVM_Pred(:,k), ~, Lin_SVM_scores(:,k)] = svmpredict(Te.y, Te.normX, Lin_cls);

    %% Quadratic kernel SVM classification
    fprintf('Training quadratic kernel SVM....\n');
    Quad_cls = svmtrain(Tr.y, Tr.normX, '-t 1 -d 2');
    fprintf('Predicting for quadratic kernel SVM....\n');
    [Quad_SVM_Pred(:,k), ~, Quad_SVM_scores(:,k)] = svmpredict(Te.y, Te.normX, Quad_cls);

    %% Cubic kernel SVM classification
    fprintf('Training cubic kernel SVM....\n');
    Cub_cls = svmtrain(Tr.y, Tr.normX, '-t 1 -d 3');
    fprintf('Predicting for cubic kernel SVM....\n');
    [Cub_SVM_Pred(:,k), ~, Cub_SVM_scores(:,k)] = svmpredict(Te.y, Te.normX, Cub_cls);

    %% Radial-basis-function kernel SVM classification
    fprintf('Training radial-basis-function kernel SVM....\n');
    RBF_cls = svmtrain(Tr.y, Tr.normX, '-t 2');
    fprintf('Predicting for radial-basis-function kernel SVM....\n');
    [RBF_SVM_Pred(:,k), ~, RBF_SVM_scores(:,k)] = svmpredict(Te.y, Te.normX, RBF_cls);

    %% Sigmoid SVM classification
    fprintf('Training sigmoid kernel SVM....\n');
    Sigm_cls = svmtrain(Tr.y, Tr.normX, '-t 3');
    fprintf('Predicting for sigmoid kernel SVM....\n');
    [Sigm_SVM_Pred(:,k), ~, Sigm_SVM_scores(:,k)] = svmpredict(Te.y, Te.normX, Sigm_cls);

end

%% Scale the scores
scale = 2 / (max( max( Lin_SVM_scores ) ) - min( min( Lin_SVM_scores ) ));
Lin_SVM = Lin_SVM_scores * scale - max( max( Lin_SVM_scores ) ) * scale + 1;

scale = 2 / (max( max( Quad_SVM_scores ) ) - min( min( Quad_SVM_scores ) ));
Quad_SVM = Quad_SVM_scores * scale - max( max( Quad_SVM_scores ) ) * scale + 1;

scale = 2 / (max( max( Cub_SVM_scores ) ) - min( min( Cub_SVM_scores ) ));
Cub_SVM = Cub_SVM_scores * scale - max( max( Cub_SVM_scores ) ) * scale + 1;

scale = 2 / (max( max( RBF_SVM_scores ) ) - min( min( RBF_SVM_scores ) ));
RBF_SVM = RBF_SVM_scores * scale - max( max( RBF_SVM_scores ) ) * scale + 1;

scale = 2 / (max( max( Sigm_SVM_scores ) ) - min( min( Sigm_SVM_scores ) ));
Sigm_SVM = Sigm_SVM_scores * scale - max( max( Sigm_SVM_scores ) ) * scale + 1;

%% See prediction performance
fprintf('Plotting performance..\n');

% and plot all together, and get the performance of each
methodNames = {'Linear SVM', 'Quadratic SVM', 'Cubic SVM', 'Radial-basis-function SVM', 'Sigmoid SVM'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods_K_CV( total_Te_y > 0, [Lin_SVM,Quad_SVM,Cub_SVM,RBF_SVM,Sigm_SVM], K, true, methodNames );
