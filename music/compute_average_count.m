load songTrain
close all
nbr_users = size(Ytrain,1);
nbr_artists = size(Ytrain,2);
%% average count and std per user
indices = (Ytrain~=0);
%gm = geomean(Ytrain(indices));
gm = 1;
Ytrain(indices) = gm*log(Ytrain(indices));

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

hist(mean_user,100)

figure
plot(mean_user, '*')
hold on;
Ylow = mean_user - std_per_user(sort_indices);
Yhigh = mean_user + std_per_user(sort_indices);
for i=1:5:nbr_users
    line([i,i],[Ylow(i),Yhigh(i)]);
end

%% average count and std per artist
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
figure
hist(mean_artist, 1000)
figure
plot(mean_artist, '*')
hold on;
Ylow = mean_artist - std_per_artist(sort_indices);
Yhigh = mean_artist + std_per_artist(sort_indices);
for i=1:5:nbr_artists
    line([i,i],[Ylow(i),Yhigh(i)]);
end
