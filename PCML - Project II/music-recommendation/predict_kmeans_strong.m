function Ypredicted = predict_kmeans_strong(Ytrain_new, Gtrain_new, ...
                                            Ytest_strong, Gstrong,...
                                            clusters, cluster_assignment)
% Mandatory inputs:
% Ytrain_new         : sparse matrix of true listening counts of 
%                      size nbr_old_users x nbr_artists
% Gtrain_new         : friendship information graph of size
%                      nbr_old_users x nbr_old_users
% Ytest_strong       : sparse matrix of new users, size nbr_new_users x
%                      nbr_artists
% Gstrong            : friendship information graph of size
%                      nbr_new_users x [nbr_old_users + nbr_new_users]
% 
% clusters           : user clusters of size nbr_clusters x nbr_artists
% cluster_assignment : old users to cluster assignment indices
%
% Outputs:
% Ypredicted         : sparse matrix of size Ytest_strong containing
%                      the predicted listening counts
%


nbr_new_users = size(Ytest_strong, 1);
nbr_old_users = size(Ytrain_new, 1);
Ypredicted = zeros(size(Ytest_strong));

% Loop throgh the new users
for u =1:nbr_new_users
       % Find the indices of the friends and friends of friends of user u
       u_friends_indices = full(Gstrong(u,1:nbr_old_users)) == 1;
       u_friends_friends_indices = full(Gtrain_new(u_friends_indices,1:nbr_old_users)) == 1;
    
       % concatenate the users
       allfriends = [u_friends_indices; u_friends_friends_indices];
       % Special case if the user had no friends among the old users
       if (sum(u_friends_indices(:))==0) 
          % Look at the old friends of the new friends that I am friends
          % with
          u_gfriends_indices = full(Gstrong(u,nbr_old_users+1:end)) == 1;
          u_friends_friends_indices = full(Gtrain_new(u_gfriends_indices,1:nbr_old_users)) == 1;
          allfriends = u_friends_friends_indices;
          % If the user really has no friends (among the new users), 
          % use the global average, equivalent with taking all old users indices.
          if (sum(allfriends(:)) == 0)
              allfriends = 1:nbr_old_users;
          end
       end

      % take the unique users indices
      allfriends = sum(allfriends);
      allfriends(allfriends~=0) = 1;
    
      % compute the mean of the clusters
      cluster_indices = cluster_assignment(boolean(allfriends));
      Ypredicted(u,:) = mean(clusters(cluster_indices,:));
end