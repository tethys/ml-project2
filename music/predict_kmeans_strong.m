function Ypredicted = predict_kmeans_strong(Ytrain_new, Gtrain_new, ...
            Ytest_strong,Gstrong,...
        clusters, cluster_assignment)


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
                  %?assert(false)
                  allf = 1:nbr_old_users;
              end
          end
          
          allf = sum(allf);
          allf(allf~=0) = 1;
          
          cluster_indices = cluster_assignment(boolean(allf));
         % cluster_indices = unique(cluster_indices);
         % cluster_indices = mode(cluster_indices);
          Ypredicted(u,:) = mean(clusters(cluster_indices,:));
end