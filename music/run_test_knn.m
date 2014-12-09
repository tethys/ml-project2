clear;
clearvars;

[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;
[similarities, simIndices] = KNNcalculateSimilarities(Ytrain_new);

Ypredicted = zeros(size(Ytest_weak));

 for i = 1:length(dd)
     Ypredicted(dd(i), nn(i)) = KNNpredict(Ytest_weak, similarities, simIndices, dd(i), nn(i));
     fprintf('For userIndex %d and artist index %d actual:%f prediction:%f.\n', dd(i), nn(i), full(Ytest_weak(dd(i), nn(i))), Ypredicted(dd(i), nn(i)));
 end

error = RMSE(Ypredicted, Ytest_weak)
