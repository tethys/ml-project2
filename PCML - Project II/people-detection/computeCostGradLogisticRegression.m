%% Normal logistic shoudl be run with lambda = 0
%% When lambda > 0 it is penalizedLogisticRegression
function [cost, grad] = computeCostGradLogisticRegression(y, tX, beta, lambda)
    N = length(y);
    %% Make sure beta(1) is not included in the penalized version of logistic cost and gradient
    cost = 1/N* (-y'*log(sigmoid(tX*beta)) - (1-y)'*log(1-sigmoid(tX*beta)))+ ...
                lambda/(2*N)*sum(beta(2:end).^2);

    beta_n = [0; beta(2:end)];
    grad = 1/N* tX'*(sigmoid(tX*beta) - y) + lambda/N*beta_n;

end