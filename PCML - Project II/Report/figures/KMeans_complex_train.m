function [clusters,cluster_assignment, train_error] = KMeans_complex_train(Ytrain, C, maxIters)

nbr_users = size(Ytrain,1);
nbr_artists = size(Ytrain,2);
nzindices = (Ytrain~=0);
Ytrain_initial = Ytrain;
global_average = sum(Ytrain(nzindices))/nnz(Ytrain);
ba = (sum(Ytrain,1))./(sum(nzindices,1) +eps) - global_average;
repba = repmat(ba, [nbr_users,1]);
zindices = Ytrain==0;
repba(zindices) = 0;
bu = (sum(Ytrain,2) - sum(repba,2))./(sum(nzindices,2) +eps) - global_average;
for u=1:nbr_users
    for a=1:nbr_artists
        if Ytrain(u,a) == 0
            Ytrain(u,a) = global_average + bu(u) + ba(a);
        end
    end
end
[S, ~] = max(Ytrain,[],2);
[~, J] = max(S);
% max(Ytrain(:))
% max(Ytrain(J,:))
[~,sorted_indices] = sort(bu);
sorted_indices
xtemp = 1:floor(nbr_users/C):nbr_users;
clusters = Ytrain(sorted_indices(xtemp(1:C)), :);
cluster_assignment = zeros(nbr_users,1);
for i=1:maxIters
    %% assign users to clusters
    for u =1:nbr_users
        differences = clusters- repmat(Ytrain(u,:),[C,1]);
        nzindices = Ytrain(u,:)~=0;
        nzcount = nnz(Ytrain(u,:));
        sumdiff = sum(differences(nzindices).^2,2)./(nzcount+eps);
        [~,cluster_index] = min(sumdiff);
        cluster_assignment(u) = cluster_index;
    end
    %% recompute clusters
    for c=1:C
        c_indices = cluster_assignment==c;
        temp = Ytrain(c_indices,:);
        nzindices = temp~=0;
        nzcount = sum(temp(nzindices));
        clusters(c,:) = sum(temp(nzindices),1)./(nzcount + eps);
    end
    train_error = computeRMSE_error(Ytrain_initial, clusters, cluster_assignment);
    %% compute train error;
    fprintf('train error %f\n', train_error);
     hist(cluster_assignment)
     pause
     close
end
train_error = computeRMSE_error(Ytrain_initial, clusters, cluster_assignment);
%% compute train error;
fprintf('train error %f\n', train_error);

end

function cost = computeRMSE_error(Ytrain_initial,clusters, cluster_assignment)
    nzindices = Ytrain_initial ~=0;
    cost = 0;
    nbr_users = size(Ytrain_initial,1);
    for u =1:nbr_users
        cindex = cluster_assignment(u);
        cost = cost + sum(exp(clusters(cindex,nzindices(u,:)) - Ytrain_initial(u,nzindices(u,:))).^2);
    end
    cost = sqrt(cost/nnz(Ytrain_initial));
end