load songTrain.mat

[ni,nj,s] = find(Gtrain);
N = length(ni);


fileID = fopen('exp.csv','w');
for i=1:N
        fprintf(fileID,'a%d; a%d\n', ni(i),nj(i));
end
fclose(fileID);

