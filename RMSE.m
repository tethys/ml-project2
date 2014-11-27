%%
%% Ypredicted, Ygroundtruth have size DxN
%%
function error = RMSE(Ypredicted, Ygroundtruth)

nonz = nonzeros(Ypredicted);
temp = abs(Ypredicted - Ygroundtruth);
error =  sum(temp(:))/ length(nonz);

end