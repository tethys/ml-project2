function nnPred = PredictNN_complex(Tr, Te, learning_rate,layers)
arr = [size(Tr.X,2) layers 2]
nn = nnsetup(arr);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 50;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 0; 
opts.dropout = 0;
nn.learningRate = learning_rate;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

% prepare labels for NN
LL = [1*(Tr.y>0)  1*(Tr.y<0)];  % first column, p(y=1)
                                   % second column, p(y=-1)
fprintf('Training\n');
[nn, L] = nntrain(nn, Tr.normX, LL, opts);

Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

fprintf('Testing\n');
% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPred = nn.a{end};

% we want a single score, subtract the output sigmoids
nnPred = nnPred(:,1) - nnPred(:,2);
