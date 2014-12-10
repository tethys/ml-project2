
%% 3,10,15,20,25,30,50,100
kmeantest = [3613,  3575,3572, 3573,3572,3572,  3571,3571];
kstdtest = [1159, 1167,    1168,    1168,    1168,    1168,  1168,    1168]

kmeantrain = [ 2.5929,2.1376,2.0195,2.0584,2.0195,2.0170, 1.9577, 1.8656]
kstdtrain=[0.062, 0.0868,    0.0405,    0.0607 ,   0.0405  ,  0.0372 , 0.0392 ,   0.0340]
tick_labels = [ '3  ';'10 ';'15 ';'20 ';'25 ';'30 ';'50 ';'100'];
figure;
plot(kmeantrain,'-*r');
hold on
Ylow = kmeantrain - kstdtrain;
Yhigh = kmeantrain + kstdtrain;
for i=1:8
    line([i,i],[Ylow(i),Yhigh(i)]);
end
set(gca,'XTickLabel',tick_labels)
print -dpdf kmean_train.pdf

figure;
plot(kmeantest,'-*r');
hold on
Ylow = kmeantest - kstdtest;
Yhigh = kmeantest + kstdtest;
for i=1:8
    line([i,i],[Ylow(i),Yhigh(i)]);
end
set(gca,'XTickLabel',tick_labels)
print -dpdf kmeans_test.pdf