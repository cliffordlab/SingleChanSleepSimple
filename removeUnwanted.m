function [features, labels, unModLabels, subjectList] = removeUnwanted(features, labels, unModLabels, subjectList, removeTransitions, sleepStages)
% Removes samples marked as 'undefined' or NaN, and removes epochs that are
% on the boarder between sleep stages, if specified.

if (removeTransitions)
    % Remove data before and after sleep stage switches from one to another
    transitionPoints = find(diff(unModLabels) ~= 0); % Find points where where sleep stage changes
    keepAndRemove = zeros(1,length(labels)); % Boolean array of which points to keep or remove
    for i = 1:length(transitionPoints)
        % Mark regions 30s before and after transition points for removal
        keepAndRemove((transitionPoints(i)-30):(transitionPoints(i)+30)) = 1;
    end
    keepAndRemove = keepAndRemove(1:length(unModLabels)); % Remove any extra points
    indicesOfPointsToRemove = find(keepAndRemove);
    labels(indicesOfPointsToRemove) = nan; % Mark transition samples for removal
end


labels(unModLabels == sleepStages('undefined'),:) = nan; % Mark undefined features for removal
labels(any(isnan(features), 2), :) = nan; % Mark features containing NaN values for removal 

subjectList = adjustForRemovedSamples(subjectList,isnan(labels)); % Modify indices of where data from each subject begins to account for feature removal

% Remove features with NaN
features(isnan(labels), :) = [];
unModLabels(isnan(labels)) = [];
labels(isnan(labels)) = [];

if any(isnan(features))
    error('Not all NaN''s removed from features')
end
if (length(labels) ~= length(features))
    error('Labels and features are not the same size.')
end
end