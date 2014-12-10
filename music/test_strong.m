[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

nbr_new_users = size(Ytest_strong, 1);
size(Gstrong)
nbr_old_users = size(Ytrain_new, 1);
%% find friends of every user in Ytest_strong

nzindices = Ytrain_new~=0;
sum_per_user = sum(Ytrain_new,2);
sum_one_per_user = sum(nzindices,2);
mean_user = sum_per_user./sum_one_per_user;
Ypredicted = repmat(mean_user, [1, size(Ytrain_new,2)]);
temp = Ypredicted;

Ypredicted = zeros(size(Ytest_strong));
for u =1:nbr_new_users
    u
   u_friends_indices = full(Gstrong(u,1:1597)) == 1;
   u_friends_friends_indices = full(Gtrain_new(u_friends_indices,1:1597)) == 1;
   allf = [u_friends_indices; u_friends_friends_indices];
   allf = sum(allf);
   allf(allf~=0) = 1;
   if (sum(u_friends_indices(:))==0)
       u_friends_indices = 1:1597;
       allf = 1:1597;
   end
   if u==156
       size(u_friends_indices)
   end
    Ypredicted(u,:) = mean(temp(boolean(allf),:));   
 %  Ypredicted(u,:) = mean(temp(u_friends_indices,:));
   ind = isnan(Ypredicted);
   sum(ind(:))
end
ind = isnan(Ypredicted);
sum(ind(:))
error = RMSE(Ypredicted, Ytest_strong)


