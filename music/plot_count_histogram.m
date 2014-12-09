clear all
close all
load songTrain;

% print some basic statistics
nbr_artists = size(Ytrain,2);
nbr_users = size(Ytrain,1);
fprintf('number of artists %d\n', nbr_artists);
fprintf('number of users %d\n', nbr_users);
fprintf('number of training entries %d\n', nnz(Ytrain));
fillcount = nnz(Ytrain)/(nbr_artists*nbr_users);
fprintf('percentage of fill %f\n', fillcount)

figure
nonzvalues = Ytrain(Ytrain~=0);
hist(nonzvalues,1000)
hx =xlabel('Histogram of listening counts');
set(hx, 'fontsize',14,'fontname','avantgarde','color',[.3 .3 .3]);
print -dpdf histYtrain.pdf

figure
nonzerovalues2 = log(nonzvalues);
hist(nonzerovalues2,1000)
hx = xlabel('Histogram of log of listening counts');
set(hx, 'fontsize',14,'fontname','avantgarde','color',[.3 .3 .3]);
print -dpdf histLogYtrain.pdf

