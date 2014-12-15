
clear all;
load songTrain;
load songTestPairs;
nbr_clusters = 20;
meanTrainMAE = 0;
meanTestMAE = 0;

maxIters = 5;
Ytrain(Ytrain~=0) = log(Ytrain(Ytrain~=0));
[clusters, cluster_assignment, train_error] = KMeansNormal_train(full(Ytrain), ...
                                           nbr_clusters, ...
                                           maxIters);
Ytest_weak_pred = zeros(size(Ytrain));
for u=1:size(Ytrain,1)
    Ytest_weak_pred(u,:) = clusters(cluster_assignment(u),:);
end
Ytest_weak_pred(Ytest_weak_pairs==0) = 0;
Ytest_weak_pred(Ytest_weak_pred~=0) = exp(Ytest_weak_pred(Ytest_weak_pred~=0));

Ypredicted = predict_kmeans_strong(Ytrain, Gtrain, ...
            Ytest_strong_pairs,Gstrong,...
            clusters, cluster_assignment);
Ytest_strong_pred = Ypredicted;
Ytest_strong_pred(Ytest_strong_pairs== 0) = 0;

Ytest_strong_pred(Ytest_strong_pairs) = exp(Ytest_strong_pred(Ytest_strong_pairs));

fprintf('finish\n');
save('songPred', 'Ytest_strong_pred', 'Ytest_weak_pred');

testMyPredSong