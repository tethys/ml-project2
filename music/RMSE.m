%%
%% Ypredicted, Ygroundtruth have size DxN
%%
function error = RMSE(Ypredicted, Ygroundtruth)

nzindices = Ygroundtruth~= 0;
temp = (Ypredicted(nzindices) - Ygroundtruth(nzindices)).^2;
sum(temp(:))
%hist(log(temp),10);
%pause
%close
error =  sqrt(sum(temp(:))/ nnz(Ygroundtruth));

end