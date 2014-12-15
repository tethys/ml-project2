function  s = sigmoid( x )
%
% Sigmoid computation
%
% Inputs:
% x  : vector on which we want to compute the sigmoid
%
% Outputs:
% s  : the resulting sigmoid of the input
%

  s = 1.0 ./ ( 1.0 + exp( -x ));

end