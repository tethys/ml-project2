function  beta = penLogisticRegression ( varargin )
%
% 'beta' computation for the method logistic regression
%
% Mandatory inputs:
% y      : y of the training dataset
% tX     : X of the training dataset
% alpha  : method parameter alpha
% lambda : method parameter lambda
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
		  lambda = 1e-7;
      case 3
          y = varargin{1};
          tX = varargin{2};
          alpha = varargin{3};
		  lambda = 1e-7;
	  case 4
          y = varargin{1};
          tX = varargin{2};
          alpha = varargin{3};
		  lambda = varargin{4};
      otherwise
          error('Unexpected number of input arguments');
  end
  
  % Initialize algorithm parametes
  maxIters = 1000;
  beta = zeros( size(tX, 2), 1 );
  err = 1 ./ eps;
  [Lold, ~] = computeCostGradLogisticRegression( y, tX, beta, lambda );
  k = 1;
  
  % Gradient descent iteration
  while ( k <= maxIters && err > eps )
        
    % Gradient computation
    [~, grad] = computeCostGradLogisticRegression( y, tX, beta, lambda );
    
    % Updating for value of 'beta'
    beta = beta - alpha .* grad;
    
    % Cost computation
    [cost, ~] = computeCostGradLogisticRegression( y, tX, beta, lambda );    
    err = abs(Lold - cost);
    Lold = cost;
    
    k = k + 1;
    
  end
  
end