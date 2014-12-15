
clear all;
load songTrain;
nbr_iterations = 2;
K = 10;
NC = 1;
meanTestMAE = zeros(nbr_iterations,K);


for i = 1:nbr_iterations
    for kfold_iter = 1:K
        fprintf('kfold %d, iter %d\n',kfold_iter, i);
        %% Do cross validation here
        seed_value = i;
        [Ytest_weak, Ytrain_new, Gtrain_new, ...
            Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
            kfold_iter, K);
        %% find friends of every user in Ytest_strong

        maxIters = 5;
            
        Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
        [clusters, cluster_assignment, train_error] = KMeansNormal_train(full(Ytrain_new), ...
                                           20, ...
                                           maxIters);
       Ypredicted = predict_kmeans_strong(Ytrain_new, Gtrain_new, ...
            Ytest_strong,Gstrong,...
        clusters, cluster_assignment);
        
        nindices = Ytest_strong~=0;
        Ytest_strong(nindices) = log(Ytest_strong(nindices));
        test_error = MAE(Ypredicted, Ytest_strong);
        meanTestMAE(i, kfold_iter) = test_error;
       fprintf('iterations %f \n', test_error)
    end
end
save('train_test_mae_strong_20kmeans_temp.mat', 'meanTestMAE')
