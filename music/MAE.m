%%
%% Ypredicted, Ygroundtruth have size DxN
%%
function error = MAE(Ypredicted, Ygroundtruth)

nzindices = Ygroundtruth~= 0;
temp = abs(Ypredicted(nzindices) - Ygroundtruth(nzindices));
sum(temp(:))
hist(temp,10);
pause
close
error = (sum(temp(:))/ nnz(Ygroundtruth));

end