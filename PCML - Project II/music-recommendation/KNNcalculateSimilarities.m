function [similarities, simIndices] = KNNcalculateSimilarities(R)
%
% This method calculates similarity matrix
% INPUT:
% R : user-artist matrix (nxm).
%
    userCount = size(R, 1);
    % initialize similarity and similarity indices matrix
    similarities = zeros(userCount, userCount);
    simIndices = zeros(userCount, userCount);
    
    % calculate similarity of user i and user j
    for i = 1:userCount-1
        similarities(i,i) = -1;
        for j = i+1:userCount
           if i ~= j
               similarities(i,j) = KNNpearsonSimilarity(R, i, j);
           end
           similarities(j,i) = similarities(i,j);
         end
    end
    
    % get sorted similarity and similarity indices matrix
    for i = 1:userCount
        [similarities(i,:), simIndices(i,:)] = sort(similarities(i,:), 'descend');
    end
end