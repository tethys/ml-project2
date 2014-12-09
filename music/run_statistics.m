
%% Look at initial data distribution, do whitening
load songTrain;

% print some basic statistics
nbr_artists = size(Ytrain,2);
nbr_users = size(Ytrain,1);
fprintf('number of artists %d\n', nbr_artists);
fprintf('number of users %d\n', nbr_users);
fprintf('number of training entries %d\n', nnz(Ytrain));
fillcount = nnz(Ytrain)/(nbr_artists*nbr_users);
fprintf('percentage of fill %f\n', fillcount)

nonzvalues = Ytrain(Ytrain~=0);
hist(nonzvalues)


very_large_counts = sum(sum(Ytrain >5000));
fprintf('very_large_counts %f\n', full(very_large_counts));
Ytrain(Ytrain >40000) = 40000;
indices = (Ytrain~=0);
max(Ytrain(:))
sum_per_artist = sum(Ytrain,1);
sum_one_per_artist = sum(indices,1);
mean_artist = sum_per_artist./sum_one_per_artist;
[mean_artist, sort_indices] = sort(mean_artist,'descend');
std_per_artist = zeros(1, nbr_artists);

for i=1:nbr_artists
    temp = Ytrain(:,i);
    %% non zero elements
    temp = temp(temp~=0);
    std_per_artist(1,i) = std(temp);

end
close all
figure
plot(mean_artist, '*')
hold on;
Ylow = mean_artist - std_per_artist(sort_indices);
Yhigh = mean_artist + std_per_artist(sort_indices);
for i=1:50:nbr_artists
    line([i,i],[Ylow(i),Yhigh(i)]);
end



% print some basic statistics
nbr_artists = size(Ytrain,2);
nbr_users = size(Ytrain,1);
fprintf('number of artists %d\n', nbr_artists);
fprintf('number of users %d\n', nbr_users);
fprintf('number of training entries %d\n', nnz(Ytrain));
fillcount = nnz(Ytrain)/(nbr_artists*nbr_users);
fprintf('percentage of fill %f\n', fillcount)


sum_per_user = sum(Ytrain,2);
sum_one_per_user = sum(indices,2);
mean_user = sum_per_user./sum_one_per_user;
[mean_user, sort_indices] = sort(mean_user,'descend');
std_per_user = zeros(nbr_users, 1);

for i=1:nbr_users
    temp = Ytrain(i,:);
    %% non zero elements
    temp = temp(temp~=0);
    std_per_user(i,1) = std(temp);
end
close all
figure
plot(mean_user, '*')
hold on;
Ylow = mean_user - std_per_user(sort_indices);
Yhigh = mean_user + std_per_user(sort_indices);
for i=1:5:nbr_users
    line([i,i],[Ylow(i),Yhigh(i)]);
end

%% Try mean and std per user. Undo later
%% Try log transform
%% Look how I compare with the rating of my friends !!!


%% Look at the current errors


%% Sunday

%% Repeat the number of features from 5-10-15,20-25-30
%% Repeat with small  0.1,0.5,0.01, 0.05
%%
%% Repeat every experiment 10 times, 
%%            do cross validation -> mean train, mean test
%% mean train, mean test, std train, std test
%%

%% Implement K means
%% Implement iterated svd