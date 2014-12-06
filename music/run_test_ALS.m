[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

[U, A] = RecomALS(full(Ytrain_new), 50, 0.06);

Ypredicted = U * A;
error = RMSE(Ypredicted, Ytest_weak)