function [U, A, Eold] = RecomExponentialALS(R, k, lambda, maxIters)
%
% Alternating Least-Squares (ALS) algorithm
% INPUT:
% R : user-artist matrix (nxm).
% k - number of hidden features
%
    
% initialize matrix U(nxk), matrix A(kxm)
[n, m] = size(R);
 rng('default')
 U = randn(n,k);
 A = randn(k,m);

% Initialize algorithm parametes
  s = 1;
  err = 1 ./ eps;
  
% calculate cost  
  Eold = calcCost(R, U, A);
  
  while ( s <= maxIters && err > eps )
      
      % Update matrices A and U
	  U = updateU(R, A, k, lambda); 
	  A = updateA(R, U, k, lambda);
	 
      % Error computation
      E = calcCost(R, U, A);
      err = abs( Eold - E );
      Eold = E;
    
      s = s + 1;
  end
end
%% Mean Absolute Error Calculation
function cost = calcCost(R, U, A)
   pp = U*A;
   nz_indices = find(R ~= 0);
   cost = sum((abs(R(nz_indices) - pp(nz_indices))));
   cost = cost / nnz(R);
end

%% Fix Matrix U and Calculate new matrix A
% k - number of hidden features
function A = updateA(R, U, k, lambda)
    % get number of artists (m)
	[~,m] = size(R);
	lamI = lambda * eye(k);
    % initialize new matrix A
    A = zeros(k, m);
    for i = 1:m
        % find non-zero indices of users for artist i
		users = find(R(:,i));
        % get non-zero user vector
		Ui = U(users, :);  
        % update A
		A(:, i) = pinv(Ui'*Ui + length(users) * lamI) * Ui' * full(R(users, i));
    end
end

%% Fix Matrix A and Calculate new matrix U
% k - number of hidden features
function U = updateU(R, A, k, lambda)
	% get number of users (n)
	[n,~] = size(R);
	lamI = lambda * eye(k);
    % initialize new matrix U
    U = zeros(n, k);
	for i = 1:n
        % find non-zero indices of artists for user i
		artists = find(R(i,:));
        % get non-zero artist vector
		Ai = A(:, artists);
        % update U
        U(i, :) = full(R(i, artists)) * Ai' * pinv(Ai*Ai' + length(artists) * lamI);
    end
end