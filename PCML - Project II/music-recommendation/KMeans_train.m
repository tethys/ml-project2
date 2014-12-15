function [clusters,cluster_assignment, train_error] = KMeans_train(Ytrain, C, maxIters)
% Mandatory inputs:
% Ytrain   : sparse matrix of true listening counts, size is
%            nbr_users x nbr_artists
% C        : number of user clusters
% maxIters : number of cluster update steps
%
% Outputs:
% clusters           : the C clusters per users of size nbr_clusters x
%                      nbr_artists
% cluster_assignment : an array of cluster indices of size nbr_users x nbr_clusters
% train_error        : mean absolute error of the non zero entries
%

% Initialize the unknown zero entries in Ytrain with the mean user score.
nbr_users = size(Ytrain,1);
nbr_artists = size(Ytrain,2);
nzindices = (Ytrain~=0);
Ytrain_initial = Ytrain;
btemp = (sum(Ytrain,2))./(sum(nzindices,2) +eps); 
for u=1:nbr_users
    for a=1:nbr_artists
        if Ytrain(u,a) == 0
            Ytrain(u,a) = btemp(u);
        end
    end
end

% Sort the users according to their average listening count
[~,sorted_indices] = sort(btemp);
xtemp = 1:floor(nbr_users/C):nbr_users;
% Inialize the clusters with equally spaced users in the sorted users array
clusters = Ytrain(sorted_indices(xtemp(1:C)), :);
% Iin the begining we have no assignment of users to clusters.
cluster_assignment = zeros(nbr_users,1);

for i=1:maxIters
    % assign users to clusters using Euclidean distance metric
    for u =1:nbr_users
        differences = clusters- repmat(Ytrain(u,:),[C,1]);
        sumdiff = sum(differences.^2,2);
        [~,cluster_index] = min(sumdiff);
        cluster_assignment(u) = cluster_index;
    end
    % recompute clusters as the mean of the assigned users
    for c=1:C
        c_indices = cluster_assignment==c;
        clusters(c,:) = sum(Ytrain(c_indices,:),1)/(sum(c_indices) + eps);
    end
    train_error = computeMAE_error(Ytrain_initial, clusters, cluster_assignment);
    %% compute train error;
    fprintf('train error %f\n', train_error);
  %   hist(cluster_assignment)
  %   pause
  %   close
end
train_error = computeMAE_error(Ytrain_initial, clusters, cluster_assignment);
fprintf('Final train error %f\n', train_error);

end

function cost = computeMAE_error(Ytrain_initial, clusters, cluster_assignment)
% Mandatory inputs:
% Ytrain_initial     : sparse matrix of true listening counts
% clusters           : computed clusters
% cluster_assignment : matrix of size nbr_usersxnbr_clusters
%
% Outputs:
% cost           : mean average error of the non zero entries in
%                   Ytrain_initial
%
    nzindices = Ytrain_initial ~=0;
    cost = 0;
    nbr_users = size(Ytrain_initial,1);
    for u =1:nbr_users
        cindex = cluster_assignment(u);
        % The error cost for user index u
        cost = cost + sum(abs((clusters(cindex,nzindices(u,:))) - (Ytrain_initial(u,nzindices(u,:)))));
    end
    % Divide by the total number of users
    cost = (cost/nnz(Ytrain_initial));
end