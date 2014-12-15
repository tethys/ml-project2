function Ypredicted = predict_kmeans_strong(Ytrain_new, Gtrain_new, ...
                                            Ytest_strong,Gstrong,...
                                            clusters, cluster_assignment)
% Mandatory inputs:
% Ytrain   : sparse matrix of true listening counts
% C        : number of listening counts
% maxIters : number of cluster update steps
%
% Outputs:
% clusters           : the C clusters per users of size nbr_clusters x
%                      nbr_artists
% cluster_assignment : an array of indices of size nbr_users x nbr_clusters
% train_error        : mean absolute error of the non zero entries
%


nbr_new_users = size(Ytest_strong, 1);
nbr_old_users = size(Ytrain_new, 1);
Ypredicted = zeros(size(Ytest_strong));
for u =1:nbr_new_users
          %  u
           u_friends_indices = full(Gstrong(u,1:nbr_old_users)) == 1;
           u_friends_friends_indices = full(Gtrain_new(u_friends_indices,1:nbr_old_users)) == 1;

           allf = [u_friends_indices; u_friends_friends_indices];
          if (sum(u_friends_indices(:))==0)
              u_gfriends_indices = full(Gstrong(u,nbr_old_users+1:end)) == 1;
              u_friends_friends_indices = full(Gtrain_new(u_gfriends_indices,1:nbr_old_users)) == 1;
              allf = u_friends_friends_indices;
              if (sum(allf(:)) == 0)
                  allf = 1:nbr_old_users;
              end
          end
          
          allf = sum(allf);
          allf(allf~=0) = 1;
          
          cluster_indices = cluster_assignment(boolean(allf));
          Ypredicted(u,:) = mean(clusters(cluster_indices,:));
end