function error = MAE(Ypredicted, Ygroundtruth)
% Mandatory inputs:
% Ygroundtruth   : sparse matrix of true listening counts
% Ypredicted     : sparse matrix of predicted listening counts
%
% Outputs:
% error          : mean absolute error of the non zero entries
%
nzindices = Ygroundtruth~= 0;
temp = abs(Ypredicted(nzindices) - Ygroundtruth(nzindices));
error = (sum(temp(:))/ nnz(Ygroundtruth));

end