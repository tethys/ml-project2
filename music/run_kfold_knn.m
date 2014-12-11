clear all;
load songTrain;
nbr_iterations = 1;
nNeighbors = [5, 10, 15, 20, 30];
NN = 5;
K = 10;
meanTrainMAE = zeros(NN,K);
meanTestMAE = zeros(NN,K);

for kfold_iter = 1:K
		%% Do cross validation here
		seed_value = 1;
		[Ytest_weak, Ytrain_new, Gtrain_new, ...
            Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
			kfold_iter, K);
				
        Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
        Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
       [similarities, simIndices] = KNNcalculateSimilarities(full(Ytrain_new));
            
             for n_index = 1:NN                              
                [test_error, train_error] = KNNpredict(similarities, simIndices, Ytrain_new, Ytest_weak, n_index);    
                meanTrainMAE(n_index, kfold_iter) = train_error;
                meanTestMAE(n_index, kfold_iter) = test_error;
                fprintf('K: %d kFold: %d testError: %f trainError: %f\n', n_index, kfold_iter, test_error, train_error)
                save('train_test_mae_KNN.mat','meanTrainMAE', 'meanTestMAE')
             end 
			 %test_error = RMSE(Ypredicted, Ytest_weak);
end
fprintf('finished KNN\n');
save('train_test_mae_KNN.mat','meanTrainMAE', 'meanTestMAE')