function [Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitDataKFold(Ytrain, Gtrain,seed_value, ...
                                                                                            kfold_iter)

% The following script creates two kinds of test sets
% Ytest_strong contains pairs for new users to test (the non-zero entries).
% Ytest_weak contains pairs for existing users to test (the non-zero entries).
% Gstrong contains the friendship information of new users with all users
% Ytrain_new and Gtrain_new is the new training set 

%% 10-fold cross validation
% Test data for Strong generalization
% keep 10% of users for testing as 'new users'
% You should decide on your own how many new users you want to test on
setSeed(seed_value);
nbr_users = size(Ytrain,1);
idx = randperm(nbr_users);
K = 10;
nTe = floor(nbr_users/K); 
NK = nTe;
for k=1:K
     idxCV(k,:) = idx(1+(k-1)*NK:k*NK);
end

% get k'th subgroup in test, others in train
idxTe = idxCV(kfold_iter,:);
idxTr = idxCV([1:kfold_iter-1  kfold_iter+1:end],:);
idxTr = idxTr(:);

Ytrain_new = Ytrain(idxTr,:);
Ytest_strong = Ytrain(idxTe,:);
Gtrain_new = Gtrain(idxTr, idxTr);
Gstrong = Gtrain(idxTe, [idxTr' idxTe]);

% Test data for weak generalization
% Keep 10 entries per existing user as test data
[D, nbr_artists] = size(Ytrain_new);
numD = 10; % number of artists held out per user
dd = [];
nn = [];
yy = [];
for n = 1:nbr_artists
    % for every artist, find the users that listened to the artist
    On = find(Ytrain_new(:,n)~=0);
    if length(On)>2  % if there are more than 2 users
        % get 10 random of them
        ind = unidrnd(length(On),numD,1); % choose some for testing
        d = On(ind);
        % keep indices of held out user ratings
        dd = [dd; d];
        % keep corresponding artist index
        nn = [nn; n*ones(numD,1)];
        % keep corresponding count
        yy = [yy; Ytrain_new(d,n)];
    end
end
Ytest_weak = sparse(dd,nn,yy, D, nbr_artists);
% 0 out the elements that are kept as weak data
Ytrain_new(sub2ind([D nbr_artists], dd, nn)) = 0;

