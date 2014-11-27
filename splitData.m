function [Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData

% The following script creates two kinds of test sets
% Ytest_strong contains pairs for new users to test (the non-zero entries).
% Ytest_weak contains pairs for existing users to test (the non-zero entries).
% Gstrong contains the friendship information of new users with all users
% Ytrain_new and Gtrain_new is the new training set 
clear all;
load songTrain;

% Test data for Strong generalization
% keep 10% of users for testing as 'new users'
% You should decide on your own how many new users you want to test on
setSeed(1);
nbr_users = size(Ytrain,1);
idx = randperm(nbr_users);
nTe = floor(nbr_users*0.1); 
idxTe = idx(1:nTe);
idxTr = idx(nTe+1:end);
Ytrain_new = Ytrain(idxTr,:);
Ytest_strong = Ytrain(idxTe,:);
Gtrain_new = Gtrain(idxTr, idxTr);
Gstrong = Gtrain(idxTe, [idxTr idxTe]);

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

