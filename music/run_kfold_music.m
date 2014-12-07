
clear all;
load songTrain;
nbr_features = [10, 30, 50, 80, 100];
alpha = [0.01,0.06, 0.1,1];
nbr_iterations = 10;
K = 10;
NF = 6;
NL = 4;
meanTrainRMSE = zeros(NF, NL,nbr_iterations,K);
meanTestRMSE = zeros(NF, NL, nbr_iterations,K);

for f_index = 1:NF
  for a_index = 1: NL
    for i = 1:nbr_iterations
        for kfold_iter = 1:K
            %% Do cross validation here
            seed_value = i;
            [Ytest_weak, Ytrain_new, Gtrain_new, ...
                Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                kfold_iter);
            maxIters = 100;
            
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
            [U, A, train_error] = RecomExponentialALS(full(Ytrain_new), ...
                                           nbr_features(f_index),...
                                           alpha(a_index), ...
                                           maxIters);
            Ypredicted = exp(U * A);
            test_error = RMSE(Ypredicted, Ytest_weak);
            meanTrainRMSE(f_index, a_index, i, kfold_iter) = train_error;
            meanTestRMSE(f_index, a_index, i, kfold_iter) = test_error;
            fprintf('iterations %f %f\n', test_error, train_error)
        end
    end
  end
end

save('train_test_rmse.mat','meanTrainRMSE', 'meanTestRMSE')
