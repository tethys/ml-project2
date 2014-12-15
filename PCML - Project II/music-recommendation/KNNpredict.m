function [test_error, train_error] = KNNpredict(similarities, simIndices, Ytrain, Ytest, k)

    Ypredicted_train = zeros(size(Ytrain));
    for i = 1:size(Ytrain, 1)
        for j = 1:size(Ytrain, 2)
            if(Ytrain(i,j) ~= 0)
                Ypredicted_train(i,j) = KNNpredictOf(Ytrain, similarities, simIndices, i, j, k);
            end
        end
    end
    train_error = MAE(Ypredicted_train, Ytrain);

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

    neighborInds = simIndices(userIndex, :);
    kNearestNeighbours = zeros(1, k);
    
    count = 1;
    for i = 1:length(neighborInds)
       if(full(R(neighborInds(i), artistIndex)) > 0)
          kNearestNeighbours(count) = neighborInds(i);
          %fprintf('Indices: %d.\n', kNearestNeighbours(count));
          count = count + 1;
          if count > k
              break;
          end
       end
    end
    
    user = R(userIndex,:);
    nnzKNearestNeighbours = kNearestNeighbours(find(kNearestNeighbours));
    meanKNearestNeighbours = mean(full(R(nnzKNearestNeighbours, artistIndex)));
    
    nominator = 0;
    denominator = 0;
    for i = 1:length(kNearestNeighbours)
        if (kNearestNeighbours(i) ~= 0)
            similarity = similarities(userIndex, kNearestNeighbours(i));
            
            nominator = nominator + similarity * (R(kNearestNeighbours(i), artistIndex) - meanKNearestNeighbours);
            denominator = denominator + abs(similarity);
        end
    end
    
    if(denominator == 0)
        prediction = nominator;
    else
        prediction = mean(user(find(user))) + (nominator / denominator);
    end
end



       