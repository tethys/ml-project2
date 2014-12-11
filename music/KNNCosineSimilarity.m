function result = KNNCosineSimilarity(R, user1Index, user2Index)
	user1 = full(R(user1Index,:));
    user2 = full(R(user2Index,:));
    
    nz_indices1 = user1 ~= 0; 
    nz_indices2 = user2 ~= 0; 
    common_artist_inds = nz_indices1 & nz_indices2;
    
    valid_user1 = user1(common_artist_inds);
    valid_user2 = user2(common_artist_inds);
    
    if(sum(common_artist_inds) == 0)
        result = 0;
        return;
    end
            
    nominator = dot(valid_user1, valid_user2);
    denominator = norm(valid_user1) * norm(valid_user2);   
    
    result = nominator/denominator;
    
    if isnan(result)
        nominator
        denominator
        size(common_artist_inds)
        sum(common_artist_inds)
        common_artist_inds
        length(valid_user1)
        assert(false)
        result = 0;
    end

    
end