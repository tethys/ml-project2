function [clusters,cluster_assignment, train_error] = kmeans_matlab(Ytrain, C, maxIters)

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
[cluster_assignment, clusters] =kmeans(Ytrain, C);

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
