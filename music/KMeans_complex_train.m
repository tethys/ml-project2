function [clusters,cluster_assignment, train_error] = KMeans_complex_train(Ytrain, C, maxIters)

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

[~,sorted_indices] = sort(btemp);
xtemp = 1:floor(nbr_users/C):nbr_users;
clusters = Ytrain(sorted_indices(xtemp(1:C)), :);

cluster_assignment = zeros(nbr_users,1);
for i=1:maxIters
    %% assign users to clusters
    for u =1:nbr_users
        differences = clusters- repmat(Ytrain(u,:),[C,1]);
        nzindices = Ytrain_initial(u,:)~=0;
        nzcount = nnz(Ytrain_initial(u,:));
        if (nzcount == 0)
            cluster_assignment(u) = 1;
        else
        sumdiff = sum(differences(:,nzindices).^2,2)./(nzcount+eps);
        [~,cluster_index] = min(sumdiff);
        cluster_assignment(u) = cluster_index;
        end
    end
%    cluster_assignment
    %% recompute clusters
    for c=1:C
        c_indices = cluster_assignment==c;
        temp = Ytrain_initial(c_indices,:);
        nzindices = temp~=0;
        nzcount = sum(nzindices,1);
       % if (prod(nzcount) == 0)
       %     clusters(c,:) = 100*ones(size(clusters(c,:)));
       % else
        HALP = sum(temp,1)./(nzcount + eps);
        clusters(c,boolean(nzcount)) = HALP(boolean(nzcount));
        %end
    end
    train_error = computeMAE_error(Ytrain_initial, clusters, cluster_assignment);
    %% compute train error;
    fprintf('train error %f\n', train_error);
   %  hist(cluster_assignment)
   %  pause
   %  close
end
train_error = computeMAE_error(Ytrain_initial, clusters, cluster_assignment);
%% compute train error;
fprintf('train error %f\n', train_error);

end

function cost = computeMAE_error(Ytrain_initial,clusters, cluster_assignment)
    nzindices = Ytrain_initial ~=0;
    cost = 0;
    nbr_users = size(Ytrain_initial,1);
    for u =1:nbr_users
        cindex = cluster_assignment(u);
        cost = cost + sum(abs((clusters(cindex,nzindices(u,:))) - (Ytrain_initial(u,nzindices(u,:)))));
    end
    cost = (cost/nnz(Ytrain_initial));
end