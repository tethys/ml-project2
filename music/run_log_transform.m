% clear;
% [Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;
% 
% 
% nbr_users = size(Ytrain_new,1);
% nbr_artists = size(Ytrain_new,2);
% sum_per_user = sum(Ytrain_new,2);
% indices= Ytrain_new~=0;
% sum_one_per_user = sum(indices,2);
% mean_user = sum_per_user./sum_one_per_user;
% std_per_user = zeros(nbr_users, 1);
% for i=1:nbr_users
%     temp = Ytrain_new(i,:);
%     %% non zero elements
%     temp = temp(temp~=0);
%     std_per_user(i,1) = std(temp);
% end
% YM = repmat(mean_user, [1,nbr_artists]);
% Ytrain_normalized = Ytrain_new - YM;
% 
% std_per_user(std_per_user == 0)= 1;
% YS = repmat(std_per_user, [1,nbr_artists]);
% Ytrain_normalized = Ytrain_normalized./YS;
% 
% [U, A] = RecomALS(full(Ytrain_normalized), 50, 0.06);
% 
% Ypredicted = U * A;
% Ypredicted = (Ypredicted + YM).*YS;
% error = RMSE(Ypredicted, Ytest_weak)

clear;
%%TODOOO put histyogram here
[Ytest_weak, Ytrain_new, Gtrain_new, Ytest_strong,Gstrong, dd,nn] = splitData;

Ytrain_new(Ytrain_new~=0) = log(Ytrain_new(Ytrain_new~=0));
maxIters = 5;
[U, A] = RecomALS(full(Ytrain_new), 100, 0.06, maxIters);

Ypredicted = exp(U * A);
error = RMSE(Ypredicted, Ytest_weak)
