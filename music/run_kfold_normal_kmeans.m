
clear all;
load songTrain;
nbr_clusters = [20,30,40,50,100];
nbr_iterations = 2;
K = 10;
NC = 5;
meanTrainMAE = zeros(NC,nbr_iterations,K);
meanTestMAE = zeros(NC, nbr_iterations,K);

for k_index = 1:NC
    for i = 1:nbr_iterations
        for kfold_iter = 1:K
            fprintf('kfold %d, iter %d, cluster %d\n',kfold_iter, i, k_index);
            %% Do cross validation here
            seed_value = i;
            [Ytest_weak, Ytrain_new, Gtrain_new, ...
                Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                kfold_iter, K);
            maxIters = 5;
            
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));

            
            [clusters, cluster_assignment, train_error] = KMeans_complex_train(full(Ytrain_new), ...
                                           nbr_clusters(k_index), ...
                                           maxIters);
            Ypredicted = zeros(size(Ytrain_new));
            for u=1:size(Ytrain_new,1)
                Ypredicted(u,:) = clusters(cluster_assignment(u),:);
            end
            Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
            test_error = MAE(Ypredicted, Ytest_weak);
            meanTrainMAE(k_index, i, kfold_iter) = train_error;
            meanTestMAE(k_index, i, kfold_iter) = test_error;
            fprintf('iterations %f %f\n', test_error, train_error)
           
            save('train_test_rmse_kmeans_complex_run.mat','meanTrainMAE', 'meanTestMAE')
        end
    end
end
fprintf('finish\n');
save('train_test_mae_normal_kmeans_complex_run.mat','meanTrainMAE', 'meanTestMAE')
