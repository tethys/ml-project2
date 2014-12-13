
%% 3,10,15,20,25,30,50,100
clear all;
load train_test_mae_normal_kmeans;

mtr = reshape(meanTrainMAE,[7,20]);
kmeantrain = mean(mtr,2);
kstdtrain = std(mtr');
tick_labels = [ '3  ';'10 ';'20 ';'30 ';'40 ';'50 ';'100'];
figure;
trainplot = plot(kmeantrain,'-*b');
hold on
Ylow = kmeantrain - kstdtrain';
Yhigh = kmeantrain + kstdtrain';
for i=1:7
    line([i,i],[Ylow(i),Yhigh(i)],'LineWidth',3);
end
set(gca,'XTickLabel',tick_labels)

mte = reshape(meanTestMAE,[7,20]);
kmeantest = mean(mte,2);
kstdtest = std(mte');

testplot = plot(kmeantest,'-*r');
hold on
Ylow = kmeantest - kstdtest';
Yhigh = kmeantest + kstdtest';
for i=1:7
    line([i,i],[Ylow(i),Yhigh(i)], 'LineWidth',3, 'Color', 'red');
end
grid on;
legend([trainplot, testplot],'Train error', 'Test error')
xlabel('Number of clusters')
set(gca,'XTickLabel',tick_labels)
print -dpdf kmeans_train_test.pdf