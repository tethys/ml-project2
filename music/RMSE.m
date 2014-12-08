%%
%% Ypredicted, Ygroundtruth have size DxN
%%
function error = RMSE(Ypredicted, Ygroundtruth)

nzindices = Ygroundtruth~= 0;
temp = (Ypredicted(nzindices) - Ygroundtruth(nzindices)).^2;
error =  sqrt(sum(temp(:))/ nnz(Ygroundtruth));

end