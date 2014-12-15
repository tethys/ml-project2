
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
        nbr_new_users = size(Ytest_strong, 1);
        nbr_old_users = size(Ytrain_new, 1);
        %% find friends of every user in Ytest_strong
        
        nzindices = Ytrain_new~=0;
        Ytrain_new(nzindices) = log(Ytrain_new(nzindices));
%         sum_per_artist = sum(Ytrain_new,1);
%         sum_one_per_artist = sum(nzindices,1);
%         mean_artist = full(sum_per_artist)./(sum_one_per_artist+eps);
%         Ypredicted = repmat(mean_artist, [ size(Ytest_strong,1),1]);
        value = full(Ytrain_new(Ytrain_new~=0));
        meanvalue = mean(value);
        Ypredicted = ones(size(Ytest_strong))*meanvalue;
        nindices = Ytest_strong~=0;
        Ytest_strong(nindices) = log(Ytest_strong(nindices));
        test_error = MAE(Ypredicted, Ytest_strong);
        meanTestMAE(i, kfold_iter) = test_error;
       fprintf('iterations %f \n', test_error)
    end
end

save('train_test_rmse_strong_global_artist_friends.mat', 'meanTestMAE')
