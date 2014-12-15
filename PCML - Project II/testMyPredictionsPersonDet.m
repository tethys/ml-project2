
clear all;
load personPred.mat;

% exists?
if ~exist('Ytest_score', 'var')
  error('No Ytest_score vector found');
end

% check size
[nU,nA] = size(Ytest_score);
if (nU ~= 8743) && (nA ~= 1)
  error('size of Ytest_score is incorrect');
end

% make sure it contains scores and not labels
nUniqueVals = length(unique(Ytest_score(:)));
if nUniqueVals < 4
    error('Ytest_score should be scores and not a binary prediction');
end

fprintf('\nSuccessful, you can submit!\n\n');