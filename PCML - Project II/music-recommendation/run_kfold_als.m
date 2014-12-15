clear all;
load songTrain;

% define different number of hidden features
nbr_features = [20,20,30,40,50,100];
NF = 6;
% define different lambda values
nlambda = [0.01, 0.05, 0.1, 0.5, 1];
NL = 5;

nbr_iterations = 1;
K = 10;

% initialize train and test error set
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
            
            % transform train and test data
            Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
            Ytest_weak(Ytest_weak~=0) = log(Ytest_weak(Ytest_weak~=0));
            
            % learn matrices U and A
            [U, A, train_error] = RecomALS(full(Ytrain_new), ...
                                           nbr_features(f_index),...
                                           nlambda(l_index), ...
                                           maxIters);
            % calculate predicted data and test error
            Ypredicted = (U * A);           
            test_error = MAE(Ypredicted, Ytest_weak);
            
            meanTrainMAE(f_index, l_index, i, kfold_iter) = train_error;
            meanTestMAE(f_index, l_index, i, kfold_iter) = test_error;

            save('train_test_rmse_logALS_6features.mat','meanTrainMAE', 'meanTestMAE')
        end
    end
  end
end
save('train_test_mae_logALS_6features.mat','meanTrainMAE', 'meanTestMAE')
