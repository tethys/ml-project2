[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

NNZ = nnz(Ytrain_new);
total_count = sum(Ytrain_new(:));
average_value = total_count/NNZ;

Ypredicted = zeros(size(Ytest_weak));
Ypredicted(dd,nn) = average_value;

error = RMSE(Ypredicted, Ytest_weak)