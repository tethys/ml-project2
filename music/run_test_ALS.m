[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

maxIters = 10000;
[U, A] = RecomALS(full(Ytrain_new), 50, 0.06, maxIters);

Ypredicted = U * A;
error = RMSE(Ypredicted, Ytest_weak)