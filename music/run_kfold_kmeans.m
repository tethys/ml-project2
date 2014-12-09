
clear all;
load songTrain;
nbr_clusters = [20, 25, 30, 50];
nbr_iterations = 10;
K = 10;
NC = 4;
meanTrainRMSE = zeros(NC,nbr_iterations,K);
meanTestRMSE = zeros(NC, nbr_iterations,K);

for k_index = 1:NC
    for i = 1:nbr_iterations
        for kfold_iter = 1:K
            %% Do cross validation here
            seed_value = i;
            [Ytest_weak, Ytrain_new, Gtrain_new, ...
                Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                kfold_iter, K);
            maxIters = 10;
            
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
            [clusters, train_error] = KMeans_train(full(Ytrain_new), ...
                                           nbr_clusters(k_index), ...
                                           maxIters);
            
            test_error = RMSE(Ypredicted, Ytest_weak);
         %   meanTrainRMSE(f_index, a_index, i, kfold_iter) = train_error;
         %   meanTestRMSE(f_index, a_index, i, kfold_iter) = test_error;
         %   fprintf('iterations %f %f\n', test_error, train_error)
        end
    end
end

save('train_test_rmse.mat','meanTrainRMSE', 'meanTestRMSE')
