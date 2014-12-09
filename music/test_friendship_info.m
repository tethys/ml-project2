clear all
load songTrain;

%Ytrain
%% Loop over all users, for every user tell me how well my number is predicted from
%% the one of my friends
nbr_users = size(Gtrain,1);

error = 0;
for u=1:nbr_users
    %% find all artists that this user ranked
    ind_artists = find(Ytrain(u,:)~= 0);
    %% find the counts of my friends
    ind_friends = find(Gtrain(u,:)~= 0);
    ycount_of_friends = full(Ytrain(ind_friends, ind_artists));
    sum_count = sum(ycount_of_friends);
    nonzerocount = sum(ycount_of_friends~=0);
    counts = sum_count./nonzerocount;
    indices = (~isnan(counts));
    temp = full(Ytrain(u,ind_artists));
    temp(indices)
    counts(indices)
    error = error + sum((temp(indices) - counts(indices)).^2);
    error
end