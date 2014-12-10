clear all;
close all
load train_test_rmse_expALS_find_nbr_features_lambda01
%load train_test_rmse_kmeans_log_50_100
meanTrainRMSE = reshape(meanTrainRMSE,[6,20]);
mtrain = mean(meanTrainRMSE,2);
strain = std(meanTrainRMSE');
plot(mtrain);

hold on;
plot(strain);

figure
meanTestRMSE = reshape(meanTestRMSE,[6,20]);
mtest = mean(meanTestRMSE,2);
stest = std(meanTestRMSE');
plot(mtest);
hold on;
plot(stest);
