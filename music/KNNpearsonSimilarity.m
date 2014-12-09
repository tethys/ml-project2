function result = KNNpearsonSimilarity(R, user1Index, user2Index)
    user1 = full(R(user1Index,:));
    user2 = full(R(user2Index,:));
         
    user1Mean = mean(user1);
    user2Mean = mean(user2);
            
    nominator = 0;
    squareSum1 = 0;
    squareSum2 = 0;

    for i = 1:length(user1)
       nominator = nominator + (user1(i) - user1Mean)*(user2(i) - user2Mean);
       squareSum1 = squareSum1 + (user1(i) - user1Mean)^2;
       squareSum2 = squareSum2 + (user2(i) - user2Mean)^2;
    end

    denominator = sqrt(squareSum1 * squareSum2);

    result = nominator / denominator;
    if isnan(result)
        result = 0;
    end
end