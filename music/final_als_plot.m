clear all;
close all
load train_test_rmse_expALS_find_nbr_features_lambda01
%load train_test_rmse_kmeans_log_50_100
meanTrainRMSE = reshape(meanTrainRMSE,[6,20]);
mtrain = mean(meanTrainRMSE,2);
strain = std(meanTrainRMSE');
figure

tick_labels = ['10 ';'20 ';'30 ';'40 ';'50 ';'100'];

plot(mtrain,'-*b');

hold on;
% on
Ylow = mtrain - strain';
Yhigh = mtrain + strain';
for i=1:6
    line([i,i],[Ylow(i),Yhigh(i)],'LineWidth',4);
end

set(gca,'XTickLabel',tick_labels,'XTick',1:6)
print -dpdf als_train.pdf
figure
meanTestRMSE = reshape(meanTestRMSE,[6,20]);
mtest = mean(meanTestRMSE,2);
stest = std(meanTestRMSE');
plot(mtest,'-*b');
hold on;
%plot(stest);
Ylow = mtest - stest';
Yhigh = mtest + stest';
for i=1:6
   % line([i,i],[Ylow(i),Yhigh(i)], 'LineWidth',3);
end

set(gca,'XTickLabel',tick_labels, 'XTick',1:6)
print -dpdf als_test.pdf
