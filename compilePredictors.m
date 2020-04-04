function compilePredictors(saveName,varargin)
% Takes list of .mat files containing saved predictors and compiles
% predictors into a single cell array and saves it to a .mat file named
% 'saveName'.
% Intended to be called from command line.
L = length(varargin);
predictorList = cell(1,L);
for i = 1:L
    load(varargin{i},'predictor')
    predictorList{i} = predictor;
end
save(saveName,'predictorList')
end