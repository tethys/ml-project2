function  beta = logisticRegression ( varargin )
%
% 'beta' computation for the method logistic regression
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
%
% Optional inputs:
% alpha  : method parameter alpha
%
% Outputs:
% beta   : estimated coefficients of the model
%

  % Chech the arguments
  switch nargin
      case 2
          y = varargin{1};
          tX = varargin{2};
          alpha = 1e-2;
      case 3
          y = varargin{1};
          tX = varargin{2};
          alpha = varargin{3};
      otherwise
          error('Unexpected number of input arguments');
  end
  
  % Initialize algorithm parameters
  maxIters = 1000;
  beta = zeros( size(tX, 2), 1 );
  err = 1 ./ eps;
  [Lold, ~] = computeCostGradLogisticRegression ( y, tX, beta, 0 );
  k = 1;
  
  % Gradient descent iteration
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    [~, grad] = computeCostGradLogisticRegression( y, tX, beta, 0 );
    
    % Updating for value of 'beta'
    beta = beta - alpha .* grad;
    
    % Cost computation
    [cost, ~] = computeCostGradLogisticRegression( y, tX, beta, 0 );    
    err = abs(Lold - cost);
    Lold = cost;
    
    k = k + 1;
    
  end
  
end