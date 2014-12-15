function result = KNNpearsonSimilarity2(R, user1Index, user2Index)
%
% Pearson Similarity method.
% calculate similarity between two given user indices
% INPUT:
% R : user-artist matrix (nxm).
%
    % get user vectors
    user1 = full(R(user1Index,:));
    user2 = full(R(user2Index,:));
    
    % find common artist indices of two users
    nz_indices1 = user1 ~= 0; 
    nz_indices2 = user2 ~= 0; 
    common_artist_inds = nz_indices1 & nz_indices2;
    
    % get non-zero user vectors
    valid_user1 = user1(common_artist_inds);
    valid_user2 = user2(common_artist_inds);
            
    % calculate mean for both user vectors
    user1Mean = mean(valid_user1);
    user2Mean = mean(valid_user2);
         
    nominator = 0;
    squareSum1 = 0;
    squareSum2 = 0;

    % calculate similarity
    for i = 1:length(valid_user1)
        nominator = nominator + (valid_user1(i) - user1Mean)*(valid_user2(i) - user2Mean);
        squareSum1 = squareSum1 + (valid_user1(i) - user1Mean).^2;
        squareSum2 = squareSum2 + (valid_user2(i) - user2Mean).^2;
    end
    denominator = sqrt(squareSum1 * squareSum2);
    result = nominator / denominator;
    
    % if both user don't have common indices, result calculation
    % will get NaN value.
    if isnan(result)
        result = 0;
    end
end 
