function result = KNNpearsonSimilarity(R, user1Index, user2Index)
    user1 = full(R(user1Index,:));
    user2 = full(R(user2Index,:));
    
    nz_indices1 = user1 ~= 0; 
    nz_indices2 = user2 ~= 0; 
    common_artist_inds = nz_indices1 & nz_indices2;
            
    nominator = 0;
    squareSum1 = 0;
    squareSum2 = 0;
    
    if(sum(common_artist_inds) == 0)
        result = 100000;
        return;
    end
    
    valid_user1 = user1(common_artist_inds);
    valid_user2 = user2(common_artist_inds);
    
    if(sum(common_artist_inds) == 1)
        result = min(valid_user1, valid_user2) / max(valid_user1, valid_user2);
        return;
    end

    user1Mean = mean(valid_user1);
    user2Mean = mean(valid_user2);
    
    user1Std = std(valid_user1);
    user2Std = std(valid_user2);

    for i = 1:length(valid_user1)
       nominator = nominator + (valid_user1(i) - user1Mean)*(valid_user2(i) - user2Mean);
    end

    denominator = user1Std * user2Std * length(valid_user1);

    result = nominator / denominator;
    
    if isnan(result)
        nominator
        denominator
        size(common_artist_inds)
        sum(common_artist_inds)
        common_artist_inds
        length(valid_user1)
        user1Std
        assert(false)
        result = 0;
    end
end