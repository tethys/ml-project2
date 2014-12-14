% Automatically calls fastROC() to show multiple curves, one for each
% prediction vector provided.
%
% labels        NxK vector
% predictions   Nx(M*K) vector, M being the number of predictions to show
% K             number of folds in Cross Validation
% if showPlot == true => single plot with multiple curves is shown.
% legendNames is a cell list (optional) with the name to show for each
% prediction in the legend.
%
% Returns tprAtWP where each element is the tprAtWP of each prediction
% vector given as input.
%
function tprAtWP = evaluateMultipleMethods_K_CV( labels, predictions, ...
                                            K, showPlot, legendNames )

if nargin < 4
    showPlot = false;
end

if nargin < 5
    legendNames = [];
end

if size(labels,1) ~= size(predictions,1)
    error('labels and predictions must have same number of rows');
end

M = size(predictions,2) / K;

% list of plotting styles
styles = {'r','b','k','m','g','r--', 'b--', 'k--','m--','g--'};

if showPlot && (M > length(styles))
    error('Number of lines to show exceeds possible styles');
end

tprAtWP = zeros(M,K);
fpr = zeros(M,K,size(labels,1));
tpr = zeros(M,K,size(labels,1));

if showPlot
    fig1 = figure;
end

for i = 1 : M
    for j = 1 : K
        [tprAtWP(i,j),~,fpr(i,j,:),tpr(i,j,:)] = fastROC( labels(:,j), predictions(:,(K*(i-1)+j)), false, styles{i} );
    end
    fpr_av = squeeze( mean( fpr(i,:,:), 2 ) );
    tpr_av = squeeze( mean( tpr(i,:,:), 2 ) );
    if showPlot
        semilogx(fpr_av,tpr_av,styles{i},'LineWidth',2);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
    end
    hold on;
end

if showPlot && ~isempty(legendNames)
    % add tprAtWP to legend names
    tprAtWP_av = mean( tprAtWP, 2 );
    for i=1:M
        legendNames{i} = sprintf('%s: %.3f', legendNames{i}, tprAtWP_av(i));
    end
    
    legend( legendNames);
    saveas(fig1, sprintf('%s,%s',legendNames{i},'.png'));
end
