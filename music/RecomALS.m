function [U, A, Eold] = RecomALS(R, k, lambda, maxIters)
%
% Alternating Least-Squares (ALS) algorithm
% INPUT:
% R : user-artist matrix (nxm).
%
    
% TODO: initialize matrix U(nxk), matrix A(kxm)
[n, m] = size(R);
 rng('default')
 U = randn(n,k);
 indices = R(:,1)~=0;
 value = mean(R(indices,1));
 A = randn(k,m);
 %A(:,1) = value;

% Initialize algorithm parametes
  s = 1;
  err = 1 ./ eps;
  
% calculate cost  
  Eold = calcCost(R, U, A);
  fprintf('costOld : %.2f\n', Eold);
  
  while ( s <= maxIters && err > eps )
      
	  U = updateU(R, A, k, lambda); 
	  A = updateA(R, U, k, lambda);
	 
      % Error computation
      E = calcCost(R, U, A);
      err = abs( Eold - E );
      Eold = E;
      
      fprintf('cost : %.2f\n', E);
    
      s = s + 1;
  end
end

function cost = calcCost(R, U, A)
   pp = U*A;
   nz_indices = find(R ~= 0);
   cost = sum((R(nz_indices) - pp(nz_indices)).^2);
   cost = cost / size(nz_indices,1);
   cost = sqrt(cost);
end

function A = updateA(R, U, k, lambda)
	
	[~,m] = size(R);
	lamI = lambda * eye(k);
    A = zeros(k, m);
	for i = 1:m
		users = find(R(:,i)); % gives non-zero entries
		Ui = U(users, :);
		%vector = Ui' * full(R(users, i));
		%matrix = Ui' * Ui + length(users) * lamI;
		%X = matrix \ vector;       
		A(:, i) = pinv(Ui'*Ui + length(users) * lamI) * Ui' * full(R(users, i));
    end
end

function U = updateU(R, A, k, lambda)
	
	[n,~] = size(R);
	lamI = lambda * eye(k);

    U = zeros(n, k);
	for i = 1:n
		artists = find(R(i,:)); % gives non-zero entries
		Ai = A(:, artists);
		%vector = Ai * full(R(i, artists))';
		%matrix = Ai * Ai' + length(artists) * lamI;
		%X = matrix \ vector;
        U(i, :) = full(R(i, artists)) * Ai' * pinv(Ai*Ai' + length(artists) * lamI);
    end
end