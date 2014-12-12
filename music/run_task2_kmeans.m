
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

        maxIters = 5;
            
        Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
        [clusters, cluster_assignment, train_error] = KMeansNormal_train(full(Ytrain_new), ...
                                           20, ...
                                           maxIters);
        Ypredicted = zeros(size(Ytest_strong));
        for u =1:nbr_new_users
          %  u
           u_friends_indices = full(Gstrong(u,1:1597)) == 1;
           allf = [u_friends_indices];
          if (sum(u_friends_indices(:))==0)
               %% gives 1:177 size again
              u_gfriends_indices = full(Gstrong(u,1598:end)) == 1;
              u_friends_friends_indices = full(Gtrain_new(u_gfriends_indices,1:1597)) == 1;
              allf = u_friends_friends_indices;
              if (sum(allf(:)) == 0)
                  %?assert(false)
                  allf = 1:1597;
              end
          end
          
          allf = sum(allf);
          allf(allf~=0) = 1;
          
          cluster_indices = cluster_assignment(boolean(allf));
         % cluster_indices = unique(cluster_indices);
          cluster_indices = unique(cluster_indices);
          %Ypredicted(u,:) = mean(clusters(cluster_indices,:));
          Ypredicted(u,:) = mean(clusters(cluster_indices,:));
        end
        
        nindices = Ytest_strong~=0;
        Ytest_strong(nindices) = log(Ytest_strong(nindices));
        test_error = MAE(Ypredicted, Ytest_strong);
        meanTestMAE(i, kfold_iter) = test_error;
       fprintf('iterations %f \n', test_error)
    end
end

save('train_test_mae_strong_kmeans20_friends.mat', 'meanTestMAE')
