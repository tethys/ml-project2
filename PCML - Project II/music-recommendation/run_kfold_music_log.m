
clear all;
load songTrain;
nbr_features = [20,20,30,40,50,100];
nlambda = [0.05];
nbr_iterations = 1;
K = 10;
NF = 6;
NL = 1;
meanTrainMAE = zeros(NF, NL,nbr_iterations,K);
meanTestMAE = zeros(NF, NL, nbr_iterations,K);

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
            Ypredicted = (U * A);
            Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
            test_error = MAE(Ypredicted, Ytest_weak);
            meanTrainMAE(f_index, l_index, i, kfold_iter) = train_error;
            meanTestMAE(f_index, l_index, i, kfold_iter) = test_error;
            fprintf('iterations %f %f\n', test_error, train_error)
            save('train_test_rmse_logALS_6features.mat','meanTrainMAE', 'meanTestMAE')

        end
    end
  end
end
fprintf('finish\n')
save('train_test_mae_logALS_6features.mat','meanTrainMAE', 'meanTestMAE')
