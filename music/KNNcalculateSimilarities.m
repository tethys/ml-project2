function [similarities, simIndices] = KNNcalculateSimilarities(R)

    userCount = size(R, 1);
    similarities = zeros(userCount, userCount);
    simIndices = zeros(userCount, userCount);
    
    for i = 1:userCount
        fprintf('Calculating similar users to user #%d\n', i);
        
        for j = 1:userCount
           if i ~= j
               similarities(i,j) = KNNpearsonSimilarity(R, i, j);
               %fprintf('Similarity of user #%d to user #%d is %f\n', j, i, similarities(i,j));
           end
         end
    end
    
    % sort similarities
    for i = 1:userCount
        [similarities(i,:), simIndices(i,:)] = sort(similarities(i,:), 'descend');
    end
end