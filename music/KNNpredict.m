function prediction = KNNpredict(R, similarities, simIndices, userIndex, artistIndex)

    k = 4;
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
    
    nominator = 0;
    denominator = 0;
    for i = 1:length(kNearestNeighbours)
        if (kNearestNeighbours(i) ~= 0)
            similarity = similarities(userIndex, kNearestNeighbours(i));
            
           %nnzKNearestNeighbours = kNearestNeighbours(find(kNearestNeighbours));
           %full(R(nnzKNearestNeighbours, artistIndex))

            nominator = nominator + similarity * (full(R(neighborInds(i), artistIndex)) - mean(full(R(neighborInds, artistIndex))));
            denominator = denominator + abs(similarity);
        end
    end
    
    prediction = nominator / denominator;
end



       