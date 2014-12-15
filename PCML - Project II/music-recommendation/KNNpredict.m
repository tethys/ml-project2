

function [test_error, train_error] = KNNpredict(similarities, simIndices, Ytrain, Ytest, k)
%
% K-nearest Neighbors Prediction method. 
% INPUT:
% similarities : user-based sorted similarity matrix (nxn)
% simIndices : user-based sorted similarity indices matrix (nxn)
% k : number of users as nearest neighbors
%
    % Calculate train error
    Ypredicted_train = zeros(size(Ytrain));
    for i = 1:size(Ytrain, 1)
        for j = 1:size(Ytrain, 2)
            if(Ytrain(i,j) ~= 0)
                Ypredicted_train(i,j) = KNNpredictOf(Ytrain, similarities, simIndices, i, j, k);
            end
        end
    end
    train_error = MAE(Ypredicted_train, Ytrain);

    % Calculate test error
    Ypredicted_test = zeros(size(Ytest));
    for i = 1:size(Ytest, 1)
        for j = 1:size(Ytest, 2)
            if(Ytest(i,j) ~= 0)
                Ypredicted_test(i, j) = KNNpredictOf(Ytest, similarities, simIndices, i, j, k);
            end
        end
    end
    test_error = MAE(Ypredicted_test, Ytest);
end

function prediction = KNNpredictOf(R, similarities, simIndices, userIndex, artistIndex, k)
%
% Prediction method of given user index and artist index. 
% INPUT:
% similarities : user-based sorted similarity matrix (nxn)
% simIndices : user-based sorted similarity indices matrix (nxn)
% k : number of users as nearest neighbors
%

    neighborInds = simIndices(userIndex, :);
    % initialize k-neigbor matrix
    kNearestNeighbours = zeros(1, k);
    
    count = 1;
    % get k nearest neigbor indices from sorted similarity indices matrix
    for i = 1:length(neighborInds)
       if(full(R(neighborInds(i), artistIndex)) > 0)
          kNearestNeighbours(count) = neighborInds(i);
          count = count + 1;
          if count > k
              break;
          end
       end
    end
    
    % if we can't find k number of indices, there will be 0 indices in
    % matrix kNearestNeighbours, so get rid of them.
    nnzKNearestNeighbours = kNearestNeighbours(find(kNearestNeighbours)); 
    
    nominator = 0;
    denominator = 0;
    % calculate weighted sum of similarities
    for i = 1:length(kNearestNeighbours)
        if (kNearestNeighbours(i) ~= 0)
            % get similarity value between active user and neigbor user
            similarity = similarities(userIndex, kNearestNeighbours(i));
            
            nominator = nominator + similarity * (R(kNearestNeighbours(i), artistIndex) - mean(R(nnzKNearestNeighbours, artistIndex)));
            denominator = denominator + abs(similarity);
        end
    end
    
    % calculate average of user
    user = R(userIndex, :);
    user = user(find(user));
    meanOfUser = mean(user);
    
    prediction = meanOfUser + (nominator / denominator);
    % if there is not any neigbors for that user, prediction calculation
    % will get NaN value since denomiator will remain 0.
    if(isnan(prediction)==1)
        prediction = 0;
    end
end



       