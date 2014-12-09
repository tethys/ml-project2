
clear all;
load songTrain;
nbr_features = [20];
nlambda = [0.01, 0.1,0.5, 1];
nbr_iterations = 1;
K = 10;
NF = 1;
NL = 4;
meanTrainRMSE = zeros(NF, NL,nbr_iterations,K);
meanTestRMSE = zeros(NF, NL, nbr_iterations,K);

for f_index = 1:NF
  for l_index = 1: NL
    for i = 1:nbr_iterations
        for kfold_iter = 1:K
            %% Do cross validation here
            seed_value = i;
            [Ytest_weak, Ytrain_new, Gtrain_new, ...
                Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                kfold_iter, K);
            maxIters = 5;
            
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
            [U, A, train_error] = RecomExponentialALS(full(Ytrain_new), ...
                                           nbr_features(f_index),...
                                           nlambda(l_index), ...
                                           maxIters);
            Ypredicted = exp(U * A);
            test_error = RMSE(Ypredicted, Ytest_weak);
            meanTrainRMSE(f_index, l_index, i, kfold_iter) = train_error;
            meanTestRMSE(f_index, l_index, i, kfold_iter) = test_error;
            fprintf('iterations %f %f\n', test_error, train_error)
        end
    end
  end
end

save('train_test_rmse_expALS_find_lambda.mat','meanTrainRMSE', 'meanTestRMSE')
