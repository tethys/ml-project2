
clear all;
load songTrain;
nbr_iterations = 2;
K = 10;
meanTrainMAE = zeros(nbr_iterations,K);
meanTestMAE = zeros(nbr_iterations,K);


    for i = 1:nbr_iterations
        for kfold_iter = 1:K
            %% Do cross validation here
            seed_value = i;
            [Ytest_weak, Ytrain_new, Gtrain_new, ...
                Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                kfold_iter, K);
            maxIters = 5;
            
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
            
            total_count = sum(Ytrain_new(:));
            NNZ = nnz(Ytrain_new);
            average_value = total_count/NNZ;
            
 
      

nzindices = Ytrain_new~=0;
sum_per_user = sum(Ytrain_new,2);
sum_one_per_user = sum(nzindices,2);
mean_user = sum_per_user./sum_one_per_user;
            mean_user(isnan(mean_user)) = average_value;
      
Ypredicted = repmat(mean_user, [1, size(Ytrain,2)]);
            Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
            test_error = MAE(full(Ypredicted), Ytest_weak);
            train_error=0;
            meanTrainMAE(i, kfold_iter) = 0;
            meanTestMAE(i, kfold_iter) = test_error;
            fprintf('iterations %f %f\n', test_error, train_error)
            save('train_test_mae_baseline_artist.mat','meanTrainMAE', 'meanTestMAE')

        end
    end

fprintf('finish\n')
save('train_test_mae_baseline_artist.mat','meanTrainMAE', 'meanTestMAE')
