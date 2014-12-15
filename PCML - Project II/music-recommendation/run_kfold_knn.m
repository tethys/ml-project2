clear all;
load songTrain;

nbr_iterations = 1;
% define different k-neigbor values
nNeighbors = [5, 10, 15, 20, 30];
NN = 5;

K = 10;
% initialize train and test error set
meanTrainMAE = zeros(NN,K);
meanTestMAE = zeros(NN,K);

for i = 1:nbr_iterations
    for kfold_iter = 1:K
		%% Do cross validation here
		seed_value = 1;
		[Ytest_weak, Ytrain_new, Gtrain_new, ...
            Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
			kfold_iter, K);
		
        % transform train and test data
        Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
        Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
        % calculate similarity and similarity indices matrices
       [similarities, simIndices] = KNNcalculateSimilarities(full(Ytrain_new));
            
       % calculate predictions for all different k-neigbor values
       % and compute train and test errors according to the predictions
        for n_index = 1:NN                              
            [test_error, train_error] = KNNpredict(similarities, simIndices, Ytrain_new, Ytest_weak, nNeighbors(n_index));    
             meanTrainMAE(n_index, kfold_iter) = train_error;
             meanTestMAE(n_index, kfold_iter) = test_error;
             save('train_test_mae_KNN.mat','meanTrainMAE', 'meanTestMAE')
        end 
    end
end
save('train_test_mae_KNN.mat','meanTrainMAE', 'meanTestMAE')