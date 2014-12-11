function [similarities, simIndices] = KNNcalculateSimilarities(R)

    userCount = size(R, 1);
    similarities = zeros(userCount, userCount);
    simIndices = zeros(userCount, userCount);
    
    for i = 1:userCount-1
        %fprintf('Calculating similar users to user #%d\n', i);
        
        for j = i+1:userCount
           if i ~= j
               similarities(i,j) = KNNCosineSimilarity(R, i, j);
               fprintf('Similarity of user #%d to user #%d is %f\n', j, i, similarities(i,j));
           end
           similarities(j,i) = similarities(i,j);
         end
    end
    
    % sort similarities
    for i = 1:userCount
        [similarities(i,:), simIndices(i,:)] = sort(similarities(i,:), 'descend');
    end
end