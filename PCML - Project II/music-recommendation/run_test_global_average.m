[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

NNZ = nnz(Ytrain_new);
total_count = sum(Ytrain_new(:));
average_value = total_count/NNZ;
Ypredicted = zeros(size(Ytest_weak));
Ypredicted(dd,nn) = average_value;
error = RMSE(Ypredicted, Ytest_weak)


nzindices = Ytrain_new~=0;
sum_per_user = sum(Ytrain_new,2);
sum_one_per_user = sum(nzindices,2);
mean_user = sum_per_user./sum_one_per_user;
Ypredicted = repmat(mean_user, [1, size(Ytrain,2)]);
error = RMSE(Ypredicted, Ytest_weak)


sum_per_artist = sum(Ytrain_new,1);
sum_one_per_artist = sum(nzindices,1);
mean_artist = sum_per_artist./sum_one_per_artist;
Ypredicted = repmat(mean_artist, [size(Ytrain_new,1), 1]);
error = RMSE(Ypredicted, Ytest_weak)

