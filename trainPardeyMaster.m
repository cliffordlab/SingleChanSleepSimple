function predictor = trainPardeyMaster(trainingRecords,channel,workingDir,varargin)
% Master code for training and testing sleep-staging algorithm on a set of physionet records.
% Currently supports formatting conventions used by Physionet's 2018 Challenge, Cyclic Alternating
% Pattern and Sleep Heart Health Study datasets.
% Performs 5-fold cross-validation unless specified to do otherwise using the -noCrossVal switch or when
% the validation set is specified.
% 
% In the output directory, the following files will be created:
% 
% <DATE>_predictor.mat - Contains trained model. Can be reloaded and tested again using the 'savedMdls' optional argument or used
% to create ensemble model.
% <DATE>_kappaPerSubj<FOLD NUMBER>.mat - Contains list of kappa value of trained model prediction accuracy for each subject,
% <DATE>_train_confusion.fig - Confusion plot of the trained model evaluated on the training data 
% <DATE>_validation_confusion.fig - Confusion plot of the trained model evaluated on the validation data
% 
% REQUIRED INPUTS
% trainingRecords ----- (string) Comma-separated list of .mat or .edf files to extract EEG data from (ex: 'file1.mat,file2.mat').
% channel ------------- (string or cell array of strings) Name of channel (ex: 'c4a1'), or possible names of channels if
%                       same channel has multiple possible names (ex: {'c4a1','C4-A1'})
% workingDir ---------- (string) Directory to save outputs or temporary files to.
% 
% OUTPUTS
% predictor ----------- ComboNet object used for predicting sleep stages 
% 
% OPTIONAL FLAGS
% -noCrossVal --------- If this switch is present, perform single run instead of 5-fold cross-validation. If validation set not
%                       specified, use the 0th fold.
% -removeArousals ----- If this switch is present, remove epochs containing arousal events
% -noFeatureRemoval --- If this switch is present, avoid removing dropped-out features during feature extraction.
% -removeTrainTrans --- If this flag is present, remove 30s of data at both sides of the points at which the subject transitions
%                       from one sleep stage to another in the training data
% -removeValTrans ----- If this flag is present, remove 30s of data at both sides of the points at which the subject transitions
%                       from one sleep stage to another in the validation data
% 
% OPTIONAL NAME-VALUE PAIRS
% nnetNodes ----------- (int array) Array of widths of each layer in network. Only used if creating new weak learner. Default = [6]
% ensemble ------------ (string) If provided, train ensemble model of provided weak learners instead of single new neural network.
%                       Value should be name of .mat file containing cell array of ComboNet objects named
% savedFeatures ------- (string) .mat containing extracted features, labels, and list of indices in the feature/label vectors where each
%  		        patient's data begins. This .mat file is created whenever extracting features and can be re-loaded whenever
%			on later runs in order to skip the feature extraction step.
% savedMdls ----------- (string) Name of .mat file containing saved ComboNet object. If present, load the saved ComboNet object
%                       and evaluate on validation set without training.
% ARorder ------------- (int) Autoregression order used for extracting reflection coefficients with Burg's algorithm. Not used if
%                       'savedFeatures' value is given. Unused if loading saved features. Default = 10.
% validationSetChosen - (int array) Specify exactly which subjects to include in validation set as indices in the list of subjects
%                       (for example, if using 5 subjects total, passing the argument [1,3,4] would cause the first, third and
%                       fourth subjects to be used for validation while the rest are used for training). 
% removeSubjects ------ (int array) Specify subjects not to include in training or test.
% writeReport --------- (string) Specify location where confusion matrix and performance can be printed do in format
%                       '<FILE NAME>,<SHEET NAME>,<START CELL>'.
% labelFiles ---------- (string) If labels in separate directory from data files, specify label file location and extension
%                       as two separate values (ex: trainPardeyMaster(...,'labelFiles','path/labelDir',',txt')). It is
%                       that each data file has a corresponding label file with the same name
% Fs ------------------ (numeric) Sampling frequency of EEG in records. Unused if loading saved features.
% runName ------------- (string) Providing this value will cause any file that is saved to be saved into a directory within
%                       workingDir with this name.
% notationTranslator -- (string) Map sleep staging labels from any database to the labels used in the cinc 2018 dataset.
%                       Example: to map the labels used in the sleep heart health study dataset (W,REM or R,S4,S3,S2,S1)
%                       to the notation used in Cinc 2018 (awake,rem,n3,n2,n1), the following string is used:
%                       'W=awake,REM=rem,R=rem,S4=n3,S3=n3,S2=n2,S1=n1'. Notice that multiple labels from one dataset can
%                       be translated into one label used by Cinc, for example, when the labels 'REM' and 'R' are both used
%                       interchangeably to refer to REM sleep. It can also be used to deal with datasets which use R&K instead
%                       of AASM rules by mapping both stage 4 and stage 3 to the 'n3' label.


%% Setup
sleepStages = containers.Map({'n3','n2','n1','rem','awake','undefined'},[0 1 2 3 4 5]); % Mapping of standard sleep stages

if (~isempty(find(strcmp(varargin,'-noCrossVal'))))
    % If this flag is entered as an input, perform single run instead of 5-fold cross-validation
    foldList = 0;
else
    foldList = 0:4;
end


for fold = foldList
    % cross-validation
    close all
    disp(['Beginning fold ' num2str(fold)])
    clearvars -except fold sleepStages trainingRecords channel workingDir varargin % Clear everything except setup and input variables
    rng(1); % Ensure replicability
    epochLength = 30;
    
    %% Parse Inputse
    if (~isempty(find(strcmp(varargin,'nnetNodes'))))
        % Obtain vector of number of neurons in each layer of the neural
        % network
        nnetLayer = str2num(varargin{find(strcmp(varargin,'nnetNodes'))+1});
    else
        nnetLayer = 6;
    end
    
    if (~isempty(find(strcmp(varargin,'ensemble'))))
        % Load list of weak predictors, if applicable
        ensemble = true;
        load(varargin{find(strcmp(varargin,'ensemble'))+1},'predictorList');
        ensembleMethod = varargin{find(strcmp(varargin,'ensemble'))+2};
    else
        ensembleMethod = 'singleton';
        ensemble = false;
    end
    
    if (~isempty(find(strcmp(varargin,'ARorder'))))
        % Obtain order for autoregression model
        ARorder = str2num(varargin{find(strcmp(varargin,'ARorder'))+1});
    else
        ARorder = 14;
    end
    
    if (~isempty(find(strcmp(varargin,'validationSetChosen'))))
        % If validation set chosen manually, load validation subject indices
        validationSetChosen = true;
        load(varargin{find(strcmp(varargin,'validationSetChosen'))+1},'validationSubjects');
	validationSubjects = double(validationSubjects); % Ensure is double
    else
        validationSetChosen = false;
    end
    
    if (~isempty(find(strcmp(varargin,'removeSubjects'))))
        % Obtain list of subjects to remove entirely
        load(varargin{find(strcmp(varargin,'removeSubjects'))+1},'removeSubjects');
    else
        removeSubjects = [];
    end
    
    if (~isempty(find(strcmp(varargin,'writeReport'))))
        % Write Excel report of data
        reportLocationData = varargin{find(strcmp(varargin,'writeReport'))+1}; % Get data regarding location to write to
    else
        % If not specified, append empty string
        reportLocationData = '';
    end
    
    if (~isempty(find(strcmp(varargin,'-removeArousals'))))
        % If this switch is present, remove epochs containing arousal events
        removeArousals = true;
    else
        removeArousals = false;
    end
    
    if (~isempty(find(strcmp(varargin,'-noFeatureRemoval'))))
        % If this switch is present, avoid removing dropped-out or excessively
        % noisy features
        noFeatureRemoval = true;
    else
        noFeatureRemoval = false;
    end
    
    if (~isempty(find(strcmp(varargin,'balanceMethod'))))
        % Choose upsampling method: smote/nnet/subsampleEpochs
	% NOTE: This feature is under development and is not recommended
	% for general use.
        balanceMethod = varargin{find(strcmp(varargin,'balanceMethod'))+1};
    else
        % Default
        balanceMethod = 'subsampleEpochs';
    end
    
    if (~isempty(find(strcmp(varargin,'labelFiles'))))
        % If labels in separate location from data files, specify label file
        % location and extension
        % NOTE: it is assumed that the labels for each record share the name of
        % the record with a different extension
        labelFileLocation = varargin{find(strcmp(varargin,'labelFiles'))+1}; % Location of labels
        labelFileExtension = varargin{find(strcmp(varargin,'labelFiles'))+2}; % File extension for labels
    else
        % Otherwise, leave as empty strings
        labelFileLocation = '';
        labelFileExtension = '';
    end
    
    if (~isempty(find(strcmp(varargin,'Fs'))))
        % Use specified sampling frequency
        if isnumeric(varargin{find(strcmp(varargin,'Fs'))+1})
            Fs = varargin{find(strcmp(varargin,'Fs'))+1};
        else
            Fs = str2num(varargin{find(strcmp(varargin,'Fs'))+1});
        end
    else
        Fs = 200; % Defaults
    end
    
    if (~isempty(find(strcmp(varargin,'runName'))))
        % Any file that is saved will now be saved to the 'runName' directory
        mkdir([workingDir varargin{find(strcmp(varargin,'runName'))+1}])
        runName = [varargin{find(strcmp(varargin,'runName'))+1} '/'];
    else
        % If not specified, append empty string
        runName = '';
    end
    
    if (~isempty(find(strcmp(varargin,'ignore'))))
        % List files to ignore
        ignore = strsplit(varargin{find(strcmp(varargin,'ignore'))+1},',');
    else
        % If not specified, leave list empty
        ignore = {};
    end
    
    translatorMap = containers.Map();
    if (~isempty(find(strcmp(varargin,'notationTranslator'))))
        % Provide means of translating sleep staging notation from any database
        % into common notation
        stagesToTranslate = strsplit(varargin{find(strcmp(varargin,'notationTranslator'))+1},',');
        for i = 1:length(stagesToTranslate)
            stageNamePair = strsplit(stagesToTranslate{i},'=');
            translatorMap(stageNamePair{1}) = stageNamePair{2};
        end
    end
    
    sleepCategories = containers.Map({'Deep','Light','REM','Awake'},[1 2 3 4]); % Maps sleep stage categories to their numerical labels
    remap = containers.Map([sleepStages('n3'),sleepStages('n2'),sleepStages('n1'),sleepStages('rem'),sleepStages('awake')] ...
            ,[sleepCategories('Deep'),sleepCategories('Light'),sleepCategories('Light'),sleepCategories('REM'),sleepCategories('Awake')]);
    
     if (~isempty(find(strcmp(varargin,'-removeTrainTrans'))))
        % If this flag is present, remove 30s of data at both sides of the
        % points at which the subject transitions from one sleep stage to
        % another in the training data
        removeTrainTrans = true;
    else
        removeTrainTrans = false;
    end
    
    
    if (~isempty(find(strcmp(varargin,'-removeValTrans'))))
        % If this flag is present, remove 30s of data at both sides of the
        % points at which the subject transitions from one sleep stage to
        % another in the validation data
        removeValTrans = true;
    else
        removeValTrans = false;
    end
    
    if (~isempty(find(strcmp(varargin,'savedMdls'))))
        % If 'savedMdls' is present as an input, load saved perceptron and skip
        % training
        loadedSavedMdl = true;
        load(varargin{find(strcmp(varargin,'savedMdls'))+1},'predictor')
        if ~exist('predictor','var')
            % Allow compatibility with models created from earlier versions
            % of code
            load(varargin{find(strcmp(varargin,'savedMdls'))+1},'pNet')
            predictor = ComboNet(ensembleMethod,{pNet});
        end
    else
        loadedSavedMdl = false;
    end
    
    %% Data extraction
    if (~isempty(find(strcmp(varargin,'savedFeatures'))))
        % If 'savedFeatures' is present as an input, load the features and
        % labels
        recordNames = {}; % Should get overriden if in saved features, otherwise can be left as an empty set
        load(varargin{find(strcmp(varargin,'savedFeatures'))+1})
    else
        % Otherwise, extract them from eeg data
        [features, labels, unModLabels, subjectList,recordNames] = ...
            extractFeaturesFromData(trainingRecords,Fs,sleepStages,translatorMap,channel,ignore,labelFileLocation,labelFileExtension,false,noFeatureRemoval,ARorder,workingDir,runName);
        save([workingDir runName strrep(datestr(now()),':','_') '_features'],'features','labels','unModLabels','subjectList','recordNames','-v7.3')
        disp('Features extracted')
    end
    
    %% Further Preprocessing
    disp(['Begin with ' num2str(length(subjectList)) ' Subjects'])
    [features, labels, unModLabels, subjectList] = removeUnwantedSubjects(features, labels, unModLabels, subjectList, removeSubjects);
    disp(['Left with ' num2str(length(subjectList)) ' Subjects'])
    
    assert(mod(size(features,1),epochLength) == 0,'Number of samples should be divisible by epoch length')
    L = size(features,1)/epochLength;
    featureMeans = nan(size(features));
    featureStd = nan(size(features));
    for i = 0:(L-1)
        % Take means and std's of each epoch
        featureMeans((1+i*epochLength):((i+1)*epochLength),:) = repelem(mean(features((1+i*epochLength):((i+1)*epochLength),:),1),epochLength,1);
        featureStd((1+i*epochLength):((i+1)*epochLength),:) = repelem(std(features((1+i*epochLength):((i+1)*epochLength),:),1),epochLength,1);
    end
    features = [featureMeans featureStd]; % combine into new feature set
    assert(all(~isnan(features),'all'),'Should be no remaining nan elements')
    
    if validationSetChosen && ~isempty(removeSubjects)
	% If necessary, compensate for fact that some subjects in subjectList were removed by creating list of how many subjects below each subject in 
	% validationSubjects were removed, then subtracting that number from each subject in list so that each element in validationSubjects still 
	% corresponds to the index in subjectList listing the location of that subjects data
	validationSubjects = setdiff(validationSubjects,removeSubjects) % Remove specified subjects that are in validation subject list

	locationsToSubtract = zeros(1,length(validationSubjects));
	for i = 1:length(removeSubjects)
	     % Tally number of subjects below each subject in validationSubjects was removed
	     locationsToSubtract(removeSubjects(i) < validationSubjects) = locationsToSubtract(removeSubjects(i) < validationSubjects) + 1;
	end
	validationSubjects = validationSubjects - locationsToSubtract;
    elseif(~validationSetChosen)
        % Create list of 20% of subjects for validation
        validationSubjects = (fold+1):5:length(subjectList);
    end
    
    % Create indices for the beginnings and endings of training and validation data
    assert(issorted(validationSubjects),'Index list creation assumes validation subjects listed in consecutive order')
    L = length(subjectList);
    subjectList(end + 1) = length(labels)+1;
    validationSubjectList = nan(length(validationSubjects),1);
    trainingSubjectList = nan(length(subjectList) - length(validationSubjects) - 1,1);
    totalValidationSubjectData = 1;
    totalTrainingSubjectData = 1;
    for i = 1:L
        if any(i == validationSubjects)
            % If subject is in list of validation subjects, add indices to list of
            % validation subject indices
            validationSubjectList(find(isnan(validationSubjectList),1)) = totalValidationSubjectData;
            
            % Calculate the index where the next subject's data (if any) will begin
            totalValidationSubjectData = totalValidationSubjectData + length(subjectList(i):(subjectList(i+1)-1));
        else
            % Otherwise, add to list of training subject indices
            trainingSubjectList(find(isnan(trainingSubjectList ),1)) = totalTrainingSubjectData;
            
            % Calculate the index where the next subject's data (if any) will begin
            totalTrainingSubjectData = totalTrainingSubjectData + length(subjectList(i):(subjectList(i+1)-1));
        end
    end
    assert(~any(isnan(trainingSubjectList),'all'),'Training subject indices incorrectly added')
    assert(~any(isnan(validationSubjectList),'all'),'Validation subject indices incorrectly added')
    
    % Gather training and validation data into separate sets
    validationIndices = false(1,length(labels)); % Indicates samples to use for validation
    for i = 1:length(validationSubjects)
        % Get data from validation subjects in list by marking samples from first
        % sample from subject through sample before first sample of subsequent subject
        validationIndices(subjectList(validationSubjects(i)):(subjectList(validationSubjects(i)+1)-1)) = true;
    end
    
    nnetTrainingIndices = ~validationIndices; % Indicates samples to use for training neural network
    subjectList(end) = []; % Remove value at ended that was previously added
    
    % Obtain validation data
    validationFeatures = features(validationIndices,:);
    validationLabels = labels(validationIndices);
    validationUnModLabels = unModLabels(validationIndices);
    [validationFeatures,validationLabels,validationUnModLabels,validationSubjectList] = removeUnwantedFeatures(validationFeatures,validationLabels,validationUnModLabels,validationSubjectList,removeValTrans,false,sleepStages);
    
    % Obtain neural network training data
    nnetTrainingFeatures = features(nnetTrainingIndices,:);
    nnetTrainingLabels = labels(nnetTrainingIndices);
    nnetTrainingUnModLabels = unModLabels(nnetTrainingIndices);
    [nnetTrainingFeatures,nnetTrainingLabels,nnetTrainingUnModLabels,trainingSubjectList] = removeUnwantedFeatures(nnetTrainingFeatures, ...
        nnetTrainingLabels, ...
        nnetTrainingUnModLabels, ...
        trainingSubjectList, ...
        removeTrainTrans, ...
        removeArousals, ...
        sleepStages); % Remove unwanted samples
    
    
    if strcmp(balanceMethod,'nnetSmote')
        % Initial training of neural network if balancing via neural-network based
        % upsampling
        [balancedTrainingFeatures, balancedTrainingLabels] = balanceBySelectedMethod(nnetTrainingFeatures,nnetTrainingUnModLabels,sleepStages,trainingSubjectList,'subsampleSamples');
        [inputFeatures, oneHotLabels] = prepareInputsForTraining(balancedTrainingFeatures, balancedTrainingLabels, sleepStages);
        upsampleNNet = trainPerceptron(inputFeatures, oneHotLabels, 6, workingDir, runName);
        disp('Upsampling network trained')
    else
        % Use empty variable if not using nnet-based upsampling to simplify
        % code
        upsampleNNet = [];
    end
    
    
    % Balance Neural Network training set
    [balancedTrainingFeatures, balancedTrainingLabels] = ...
        balanceBySelectedMethod(nnetTrainingFeatures, ... 
        arrayfun(@(x) remap(x),nnetTrainingUnModLabels), ...
        sleepCategories, ...  
        trainingSubjectList, ... % Adjust patient start indices for removed samples
        balanceMethod);
    disp(['Reduced all NNet classes to ' num2str(length(balancedTrainingFeatures)) ' features for training.'])
    
    % Balance validation set
    remappedValidationLabels = arrayfun(@(x) remap(x), validationUnModLabels);
    [balancedValidationFeatures, actualValidationCategories] = ...
        balanceBySelectedMethod(validationFeatures,remappedValidationLabels,sleepCategories,validationSubjectList,'subsampleEpochs');
    disp('Preprocessing complete')
    
    %% Model Training
    if loadedSavedMdl
        % Use existing predictor, do nothing
        disp('Using saved predictor')
    elseif ensemble
        % Combine set of weak predictors
        % Note: training using every every 30th sample because all features
        % in single epoch are the same
        [inputFeatures, oneHotLabels] = prepareInputsForTraining(balancedTrainingFeatures(1:epochLength:end,:), balancedTrainingLabels(1:epochLength:end), sleepCategories);
        predictor = train(ComboNet(ensembleMethod,predictorList),inputFeatures,oneHotLabels);
        
        save([workingDir runName strrep(datestr(now()),':','_') '_predictor'],'predictor')
    else
        % Otherwise, train perceptron
        % Note: training using every every 30th sample because all features
        % in single epoch are the same
        [inputFeatures, oneHotLabels] = prepareInputsForTraining(balancedTrainingFeatures(1:epochLength:end,:), balancedTrainingLabels(1:epochLength:end), sleepCategories);
        
        % Construct perceptron
        pNet = patternnet(nnetLayer,'trainlm','mse');
        pNet.Layers{1:(end-1)}.transferFcn = 'logsig';
        predictor = train(ComboNet(ensembleMethod,{pNet}),inputFeatures,oneHotLabels);

        save([workingDir runName strrep(datestr(now()),':','_') '_predictor'],'predictor')
    end
    
    %% Validate Model
    
    % Obtain reflection coefficients and hidden layer outputs for patients most
    % accurately labeled by trained perceptron for use in cluster analysis
    [clusteringFeatures,~,clusteringUnModLabels,clusteringSubjectList] = removeUnwantedFeatures(nnetTrainingFeatures, ...
        nnetTrainingLabels, ...
        nnetTrainingUnModLabels, ...
        trainingSubjectList, ...
        removeTrainTrans,false,sleepStages);
    kappaPerSubj = getKappaPerSubj(clusteringFeatures,arrayfun(@(x) remap(x),clusteringUnModLabels),clusteringSubjectList,predictor);
    disp(['Saving data to ' [workingDir runName strrep(datestr(now()),':','_') '_kappaPerSubj' num2str(fold)]])
    save([workingDir runName strrep(datestr(now()),':','_') '_kappaPerSubj' num2str(fold)],'kappaPerSubj','-v7.3')
    
    % Memory management
    clear features labels unModLabels
    
    predictedTrainCategories = predictor(balancedTrainingFeatures); % Predict sleep category
    
    [C, overallStats] = stageConfusion(balancedTrainingLabels,predictedTrainCategories,sleepCategories,'Training Data');
    saveas(gcf,[workingDir runName strrep(datestr(now()),':','_') '_train_confusion'])
    if ~isempty(reportLocationData)
        % Obtain information regarding where results should be printed to, if
        % any
        reportLocationDataCells = strsplit(reportLocationData,',');
        
        filename = reportLocationDataCells{1};
        sheetName = [reportLocationDataCells{2} ' ' num2str(fold)];
        startCell = reportLocationDataCells{3};
        
        writetable(cell2table(C'),filename,'Range',startCell,'Sheet',sheetName,'WriteVariableNames',false);
        writetable(cell2table(overallStats(:,2)),filename,'Range',moveCell(startCell,0,size(C',2)+1),'Sheet',sheetName,'WriteVariableNames',false);
        if (length(reportLocationDataCells) > 3)
            % If additional data provided indicating overallStats should be
            % printed elsewhere, obtain that too
            overallFilename = reportLocationDataCells{4};
            overallSheetName = reportLocationDataCells{5};
            overallStartCell = reportLocationDataCells{6};
            writetable(cell2table(overallStats(:,2)),overallFilename,'Range',overallStartCell,'Sheet',overallSheetName,'WriteVariableNames',false);
        end
    end
    
    predictedValdiationCategories = predictor(balancedValidationFeatures); % Predict sleep category
    
    [C, overallStats] = stageConfusion(actualValidationCategories,predictedValdiationCategories,sleepCategories,'Validation Data');
    saveas(gcf,[workingDir runName strrep(datestr(now()),':','_') '_validation_confusion'])
    if ~isempty(reportLocationData)
        % Obtain information regarding where results should be printed to, if
        % any
        writetable(array2table(C'),filename,'Range',moveCell(startCell,6,0),'Sheet',sheetName,'WriteVariableNames',false);
        writetable(cell2table(overallStats(:,2)),filename,'Range',moveCell(startCell,6,size(C',2)+1),'Sheet',sheetName,'WriteVariableNames',false);
        if (length(reportLocationDataCells) > 3)
            % If additional data provided indicating overallStats should be
            % printed elsewhere, obtain that too
            writetable(cell2table(overallStats(:,2)),overallFilename,'Range',moveCell(overallStartCell,5,0),'Sheet',overallSheetName,'WriteVariableNames',false);
        end
    end
    
    % Report other sleep stage statistics
    [mdl_sol,mdl_tst,mdl_waso,mdl_se] = calcSleepStats(predictor(validationFeatures),sleepCategories,validationSubjectList,epochLength);
    [true_sol,true_tst,true_waso,true_se] = calcSleepStats(remappedValidationLabels,sleepCategories,validationSubjectList,epochLength);
    disp('Sleep Onset Latency')
    disp(['Model Average: ' num2str(mean(mdl_sol))])
    disp(['True Model: ' num2str(mean(true_sol))])
    disp(['Relative Error: ' num2str(mean((mdl_sol - true_sol)./true_sol))])
    [R,~,RL,RU] = corrcoef(mdl_sol,true_sol);
    disp(['Correlation: ' num2str(R(1,2)) ' (' num2str(RL(1,2)) ' - ' num2str(RU(1,2)) ')'])
    
    disp('Total Sleep Time')
    disp(['Model Average: ' num2str(mean(mdl_tst))])
    disp(['True Model: ' num2str(mean(true_tst))])
    disp(['Relative Error: ' num2str(mean((mdl_tst - true_tst)./true_tst))])
    [R,~,RL,RU] = corrcoef(mdl_tst,true_tst);
    disp(['Correlation: ' num2str(R(1,2)) ' (' num2str(RL(1,2)) ' - ' num2str(RU(1,2)) ')'])
    
    disp('Wake After Sleep Onset')
    disp(['Model Average: ' num2str(mean(mdl_waso))])
    disp(['True Model: ' num2str(mean(true_waso))])
    disp(['Relative Error: ' num2str(mean((mdl_waso - true_waso)./true_waso))])
    [R,~,RL,RU] = corrcoef(mdl_waso,true_waso);
    disp(['Correlation: ' num2str(R(1,2)) ' (' num2str(RL(1,2)) ' - ' num2str(RU(1,2)) ')'])
    
    disp('Sleep Efficiency')
    disp(['Model Average: ' num2str(mean(mdl_se))])
    disp(['True Model: ' num2str(mean(true_se))])
    disp(['Relative Error: ' num2str(mean((mdl_se - true_se)./true_se))])
    [R,~,RL,RU] = corrcoef(mdl_waso,true_se);
    disp(['Correlation: ' num2str(R(1,2)) ' (' num2str(RL(1,2)) ' - ' num2str(RU(1,2)) ')'])
end
end

function [confusionMat, overallStats] = stageConfusion(actualCategories,predictedCategories,sleepCategories,titleText)
% Plot confusion matrix along with accuracy and kappa values for sleep
% staging data

categoryNames = keys(sleepCategories);
[orderedValues,valueOrder] = sort(cell2mat(values(sleepCategories)));
confusionMat = cell(length(categoryNames)+2,length(categoryNames));

% Calculate Average of Individual Accuracies
C = confusionmat(actualCategories,predictedCategories,'ORDER',orderedValues);
numDiags = length(C); % Because C is square, the number of diagonals is equal to the length of any side
recalls = zeros(1,numDiags);
for i = 1:numDiags
    recalls(i) = C(i,i)/sum(C(i,:));
end
aveRecall = 100*sum(recalls)/numDiags;

% Create grid
figure
ax = gca;
ax.FontSize = 13;
ax.YDir = 'reverse';
ax.DataAspectRatioMode = 'manual';
xlabel('Actual','FontSize',14,'FontWeight','bold')
ylabel('Predicted','FontSize',14,'FontWeight','bold')
L = length(values(sleepCategories));
for i = 0:L
    for j = 0:(L-1)
        xdata = [(i+.5) (i+1.5) (i+1.5) (i+.5)];
        ydata = [(j+.5) (j+.5) (j+1.5) (j+1.5)];
        if (i == length(values(sleepCategories)))
            c = [0.94,0.94,0.94]; % Color right-edge cells grey
        elseif (i == j)
            c = [0.74,0.90,0.77]; % Color diagonal cells green
        else
            c = [0.98,0.77,0.75]; % Color remaining cells red
        end
        patch(xdata,ydata,c)
    end
end
ylim([.5 (L+.5)])
xlim([.5 (L+1.5)])
xticks(1:(L+1))
yticks(1:L)

xticklabels([categoryNames(valueOrder) 'Precision'])
yticklabels(categoryNames(valueOrder))

kappa = zeros(1,length(categoryNames));
for i = 1:length(categoryNames)
    % Calculate Cohen's Kappa for each class
    iStageLabel = sleepCategories(categoryNames{valueOrder(i)}); % Get label of category corresponding to i'th row
    predictedIsStage = (predictedCategories == iStageLabel); % Boolean indicating which values predicted to belong to category
    actualIsStage = (actualCategories == iStageLabel); % Boolean indicating which values actually belong to category
    kappa(i) = cohenKappa(predictedIsStage,actualIsStage);
end

% NOTE: elements in plot are in transposed positions to those same elements
% in 'C'
L = length(categoryNames);
precision = zeros(1,L);
for i = 1:L
    % Add text to confusion table
    for j = 1:(L+1)
        if(j == (L+1))
            % Right column
            precision(i) = 100*C(i,i)/sum(C(:,i)); % Precision
            confusionMat{j,i} = num2str(precision(i));
            t = text(j,i,[num2str(precision(i)) '%']);
            t.HorizontalAlignment = 'center';
            t.Color = [0 .4 0];
            t.FontSize = 12;
            
            % Plot kappa
            t = text(j,i+.25,['\kappa = ' num2str(kappa(i))]);
            confusionMat{j+1,i} = num2str(kappa(i));
            t.HorizontalAlignment = 'center';
            t.FontSize = 12;
        else
            % Main body
            t = text(i,j,num2str(C(i,j))); % Total samples in 'actualCategory' that were classified as 'predictedCategory'
            t.HorizontalAlignment = 'center';
            t.FontWeight = 'bold';
            t.VerticalAlignment = 'bottom';
            t.FontSize = 12;
            
            t = text(i,j,[num2str(100*C(i,j)/sum(C(i,:))) '%']); % Percent of samples in 'actualCategory' that were classified as 'predictedCategory'
            confusionMat{i,j} = sprintf([num2str(100*C(i,j)/sum(C(i,:))) '\n' num2str(C(i,j))]);
            t.HorizontalAlignment = 'center';
            t.VerticalAlignment = 'top';
            t.FontSize = 12;
        end
    end
end

title({titleText; ['Average Recall = ' num2str(aveRecall) '%' ...
    ', Average Precision = ' num2str(mean(precision)) '%']; ...
    ['Overall Accuracy = ' num2str(100*sum(predictedCategories == actualCategories)/length(actualCategories)) ...
    '%, Average \kappa = ' num2str(mean(kappa)) ...
    ', Overall \kappa = ' num2str(cohenKappa(actualCategories,predictedCategories))]},'FontSize',15)

% Return overall stats as cell array
overallStats = {'Average Recall', aveRecall; ...
    'Average Precision', mean(precision); ...
    'Overall Accuracy', 100*sum(predictedCategories == actualCategories)/length(actualCategories); ...
    'Average k', mean(kappa); ...
    'Overall k', cohenKappa(actualCategories,predictedCategories)};
end

function kappa = cohenKappa(data1,data2)
% Calculate cohen's kappa between two sets of labels
classes = union(unique(data1),unique(data2)); % Get all classes in sets
Po = sum(data1 == data2)/length(data1); % Find agreement

% Find probability of chance agreement
Pe = 0;
for i = 1:length(classes)
    Pe = Pe + sum(data1 == classes(i))*sum(data2 == classes(i));
end
Pe = Pe/length(data1)^2;
kappa = (Po - Pe)/(1 - Pe); % Calculate kappa
end

function categoryLabels = sleepDepth2hypAverageEpoch(sleepDepth,categoryDepth,sleepStageCategory)
% Assign label of deep, light, REM or wakefulness to sleepDepth and output
% hypnogram
L = length(sleepDepth);
possibleDepths = cell2mat(keys(categoryDepth));
categoryLabels = zeros(L,1);
epochLength = 30;
assert(mod(length(sleepDepth),epochLength) == 0,'Length of sleep depth array must be integer multiple of epoch length')

% reassign each sample to the average of all samples in same epoch
M = length(sleepDepth)/epochLength;
averageSleepDepthPerEpoch = zeros(length(sleepDepth),1);
for i = 1:M
    averageSleepDepthPerEpoch(((i-1)*epochLength+1):(i*epochLength)) = mean(sleepDepth(((i-1)*epochLength+1):(i*epochLength)));
end

dist2PossibleDepths = abs(averageSleepDepthPerEpoch - possibleDepths); % Distance from depth of each sample to depths of each sleep stage category

% Assign numerical label to each sample corresponding to closest sleep
% category
for i =1:(L-1)
    closestInd = find(dist2PossibleDepths(i,:) == min(dist2PossibleDepths(i,:))); % Index of value closest to sleep category depth
    categoryLabels(i) = sleepStageCategory(categoryDepth(possibleDepths(closestInd))); % Assign numerical label of closest sleep category
end
%categoryLabels = colfilt(categoryLabels, [5 1], 'sliding', @mode); % Smoothen output with mode filter

% Remove 0's resulting from padding
indFirstNonZero = find(categoryLabels ~= 0,1,'first');
categoryLabels(1:(indFirstNonZero-1)) = categoryLabels(indFirstNonZero); % Assign leading 0's to first non-0 value
indLastNonZero = find(categoryLabels ~= 0,1,'last');
categoryLabels((indLastNonZero+1):end) = categoryLabels(indLastNonZero); % Assign trailing 0's to last non-0 value
end

function categoryLabels = sleepDepth2hyp(sleepDepth,categoryDepth,sleepStageCategory)
% Assign label of deep, light, REM or wakefulness to sleepDepth and output
% hypnogram
epochLength = 30;
assert(mod(length(sleepDepth),epochLength) == 0,'Length of sleep depth should be integer multiple of epochLength')
L = length(sleepDepth)/epochLength;
possibleDepths = cell2mat(keys(categoryDepth));
categoryLabels = zeros(length(sleepDepth),1);

filteredSleepDepth = smooth(sleepDepth,30,'sgolay');
dist2PossibleDepths = abs(filteredSleepDepth - possibleDepths); % Distance from depth of each sample to depths of each sleep stage category

% Assign numerical label to each epoch corresponding to closest sleep
% category
for i =1:L
    closestInd = find(dist2PossibleDepths((i-1)*epochLength+round(epochLength/2),:) == min(dist2PossibleDepths((i-1)*epochLength+round(epochLength/2),:))); % Index of value closest to sleep category depth in center of smoothened epoch
    categoryLabels(((i-1)*epochLength+1):(i*epochLength)) = sleepStageCategory(categoryDepth(possibleDepths(closestInd))); % Assign numerical label of closest sleep category to all samples in epoch
end
end

function [features, labels, unModLabels, subjectList, recordNames] = extractFeaturesFromData(records,Fs,stages,translatorMap,channelNames,ignore,labelFileLocation,labelFileExtension,ignoreWrongSamplingRate,noFeatureRemoval,ARorder,workingDir,runName)
% Extract EEG from records, then obtain features from data
trainingRecordsList = strsplit(records,',');
L = length(trainingRecordsList);
channelList = strsplit(channelNames,',');
stdWindow = 7*Fs; % Window size of moving standard deviation used for artifact detection
thresh = 2; % Number of standard deviations above/below the mean moving standard deviation to classify as sample as a movement artifact
shhs2cinc = containers.Map([0,1,2,3,4,5],[4,2,1,0,0,3]); % Map shhs labels to those used by the Physionet challenge
tmpFileIdentifier = num2str(randi(1e6,1,1)); % Randomly generated number to identify which tmp files belong to which instance, preventing the risk of collisions
epochLength = 30; % Length of epoch in seconds

% Initialize data
features = zeros(L*8*3600,ARorder); % Initialize to larger array than necessary
labels = zeros(L*8*3600,1);
unModLabels = zeros(L*8*3600,1); % Record what epochs were labeled before re-classifying arousal regions
subjectList = nan(1,L); % Track locations of data for each subject (initialize with NaN to make it easier to find ignored values)
recordNames = cell(1,L); % Track names of each record

totalDataAdded = 0; % Keep track of amount of data added
removedFeatures = 0; % Track number of features that have been removed

for i = 1:L
    [~,fileName] = fileparts(trainingRecordsList{i});
    if any(strcmp(ignore,fileName))
        qdisp(['Ignoring record ' trainingRecordsList{i}],workingDir,runName)
        continue
    end
    qdisp(['Obtaining data from record ' trainingRecordsList{i}],workingDir,runName)
    if isfile([trainingRecordsList{i} '.mat'])
        % Get signals and label if input is .mat file
        [~,allSignals,recordFs,siginfo]=rdmat(trainingRecordsList{i}); % Get signal
        if (recordFs ~= Fs && ignoreWrongSamplingRate)
            % Skip files with wrong sampling rate, if specified to do so
            qdisp(['record ' trainingRecordsList{i} ' is using sampling rate ' num2str(recordFs) '. Skipping file.'],workingDir,runName);
            continue
        end
        
        % Get labels for same signal
        load([trainingRecordsList{i} '-arousal.mat']);
        unModLabelsFrom1Subj = extractLabels(data.sleep_stages,stages);
        arousals = data.arousals';
        
        signalIndex = find(contains({siginfo.Description},channelList), 1); % Index of desired channel in data matrix
        if (isempty(signalIndex))
            % Ensure desired channel is present in file
            qdisp(['record ' trainingRecordsList{i} ' does not have any of the following channels: ' channelNames],workingDir,runName);
            continue
        end
        signal = allSignals(:,signalIndex); % Obtain only signal from desired channel
        
    elseif (isfile(trainingRecordsList{i}) && strcmp(trainingRecordsList{i}(end-3:end),'.edf'))
        assert(~isempty(translatorMap),'notationTranslator not specified')
        
        % Copy record into working directory
        w = system(['cp ' trainingRecordsList{i} ' tmp' tmpFileIdentifier '.edf']);
        assert(w == 0,['Could not find file ' trainingRecordsList{i}])
        
        [~,siginfo] = system(['wfdbdesc tmp' tmpFileIdentifier '.edf']); % Get file description
        signalList = extractBetween(siginfo,'Description: ',newline); % Get list of signals
        signalIndex = find(contains(signalList,channelList), 1); % Index of desired channel in data matrix
        [status,textOutput] = system(['wfdb2mat -r tmp' tmpFileIdentifier '.edf -s "' signalList{signalIndex} '"']); % Convert .edf file to more easily readable .mat file
        
        if (status ~= 0)
            qdisp(['Problem converting ' trainingRecordsList{i}  ' to .mat.'],workingDir,runName)
            continue
        end
        if (isempty(signalIndex))
            % Ensure desired channel is present in file
            qdisp(['record ' trainingRecordsList{i} ' does not have any of the following channels: ' channelNames],workingDir,runName);
            continue
        end
        
        try
            signal = load(['tmp' tmpFileIdentifier '_edfm.mat'],'val'); % Get data
            signal = signal.val';
            
            recordFs = cellfun(@str2num,extractBetween(textOutput,'Sampling frequency: ',' Hz')); % Obtain sampling frequency
            
            if (recordFs ~= Fs && ignoreWrongSamplingRate)
                % Skip files with wrong sampling rate, if specified to do so
                qdisp(['record ' trainingRecordsList{i} ' is using sampling rate ' num2str(recordFs) '. Skipping file.'],workingDir,runName);
                continue
            end
            
            % TODO: simplify this section to same code with default
            % values in code parsing section of main function
            if isempty(labelFileLocation)
                % If labelFileLocations empty, assume the labels are kept
                % in an .st file with the same name as the data file
                w = system(['cp ' trainingRecordsList{i} '.st tmp' tmpFileIdentifier '.edf.st']);
                assert(w == 0,'No .st file')
                [unModLabelsFrom1Subj,arousals] = getLabelsFromAnn(['tmp' tmpFileIdentifier '.edf'],'st',recordFs,length(signal),translatorMap,stages,workingDir,runName); % Get labels
            else
                % Otherwise, assume it is a .mat file
                labelFile = [labelFileLocation '/' fileName labelFileExtension];
                labelData = load(labelFile);
                labelData.stages = arrayfun(@(x) shhs2cinc(x), labelData.stages'); % Map shhs data label interpretation to common interpretation
                unModLabelsFrom1Subj = repelem(labelData.stages,30*Fs); % Load data
                arousals = zeros(length(unModLabelsFrom1Subj),1); % No arousals specified, so leave empty
            end
            
        catch ME
            switch ME.identifier
                case 'data_extraction:temporally_inconsistent_indices'
                    qdisp('Indices suggested overlapping sleep stages. Skipping record.',workingDir,runName);
                    continue
                otherwise
                    qdisp('Unknown error parsing record. Skipping file.',workingDir,runName);
                    qdisp('Original error message:',workingDir,runName)
                    qdisp(ME.message,workingDir,runName)
                    continue
            end
        end
        
        % Skip any files in which there appears to be samples missing from
        % the EEG data
        if (length(unModLabelsFrom1Subj) ~= length(signal))
            qdisp(['Differing lengths between sleep stage annotation data length (' ...
                num2str(length(unModLabelsFrom1Subj)) ') and signal length (' num2str(length(signal)) ...
                '). Skipping record ' trainingRecordsList{i}],workingDir,runName);
            continue
        elseif (length(arousals) ~= length(signal))
            qdisp(['Differing lengths between sleep arpousal annotation data length (' ...
                num2str(length(arousals)) ') and signal length (' num2str(length(signal)) ...
                '). Skipping record ' trainingRecordsList{i}],workingDir,runName);
            continue
        end
        
        % Remove temporary files
        delete(['tmp' tmpFileIdentifier '_edfm.mat'])
        delete(['tmp' tmpFileIdentifier '_edfm.hea'])
        delete(['tmp' tmpFileIdentifier '.edf'])
        delete(['tmp' tmpFileIdentifier '.edf.st'])
    else
        error(['Could not find record ' trainingRecordsList{i}])
    end
    
    if (recordFs ~= Fs)
        % Ensure all files using same sampling frequency
        disp('Resampling')
        signal = resample(signal,recordFs,Fs);
        unModLabelsFrom1Subj = round(resample(unModLabelsFrom1Subj,recordFs,Fs));
        arousals = round(resample(arousals,recordFs,Fs));
        assert(length(signal) == length(unModLabelsFrom1Subj) && length(signal) == length(arousals),'Signal and labels should be same sizes')
    end
    
    % Shave off ends of signal so that signal duration is a whole number of seconds
    signal = signal(1:(epochLength*Fs*floor(length(signal)/(epochLength*Fs))));
    unModLabelsFrom1Subj = unModLabelsFrom1Subj(1:(epochLength*Fs*floor(length(unModLabelsFrom1Subj)/(epochLength*Fs))));
    arousals = arousals(1:(epochLength*Fs*floor(length(arousals)/(epochLength*Fs))));
    
    filteredSignal = bandpass(signal,[.5 30],Fs,'ImpulseResponse','iir','Steepness',0.95); % filter
    
    % Assign label of 'awake' to arousal regions
    labelsFrom1Subj = unModLabelsFrom1Subj;
    labelsFrom1Subj(arousals == 1) = stages('awake');
    
    % Mark regions containing artifacts or other unwanted data for later removal
    artifactRegion = false(length(filteredSignal),1);
    if (~noFeatureRemoval)
        %artifactRegion = movstd(filteredSignal,stdWindow) > thresh*std(filteredSignal); % Mark samples as artifact if moving std is greater than some multiple of the signal std
        artifactRegion = artifactRegion | movstd(filteredSignal,stdWindow) < 1e-15; % Mark samples as dropped-out/railed if signal is flat
    end
    artifactRegion = artifactRegion | isnan(unModLabelsFrom1Subj); % Mark samples where label was NaN
    %artifactRegion = artifactRegion | (arousals == -1); % Mark samples where artifact was unlabeled
    
    
    % Feature extraction
    featuresFrom1Subj = getEEGAutoregressFeatures(filteredSignal,Fs,ARorder); % Obtains one feature vector for every second of data
    labelsFrom1Subj = labelsFrom1Subj(1:Fs:end); % Downsample labels so that there is only one per feature
    unModLabelsFrom1Subj = unModLabelsFrom1Subj(1:Fs:end);
    artifactRegion = artifactRegion(1:Fs:end);
    artifactRegion = artifactRegion | any(isnan(featuresFrom1Subj),2); % Mark samples containing NaN
    
    % Mark samples belonging to epochs containing artifacts
    epochsToRemove = false(1,length(artifactRegion));
    for j = 1:floor(length(labelsFrom1Subj)/epochLength)
        if any(artifactRegion(((j-1)*epochLength+1):(j*epochLength)))
            epochsToRemove(((j-1)*epochLength+1):(j*epochLength)) = true;
        end
    end
    
    % Remove epochs that contain artifacts
    disp('Artifacts were found in samples with the following labels:')
    disp(unique(labelsFrom1Subj(epochsToRemove)))
    labelsFrom1Subj(epochsToRemove) = [];
    unModLabelsFrom1Subj(epochsToRemove) = [];
    featuresFrom1Subj(epochsToRemove,:) = [];
    removedFeatures = removedFeatures + sum(epochsToRemove);
    
    % Add newly acquired data to larger collection
    features((totalDataAdded+1):(totalDataAdded+length(featuresFrom1Subj)),:) = featuresFrom1Subj;
    labels((totalDataAdded+1):(totalDataAdded+length(labelsFrom1Subj))) = labelsFrom1Subj;
    unModLabels((totalDataAdded+1):(totalDataAdded+length(unModLabelsFrom1Subj))) = unModLabelsFrom1Subj;
    
    subjectList(i) = totalDataAdded + 1; % Record index where data of new subject begins
    recordNames{i} = fileName;
    totalDataAdded = totalDataAdded + length(featuresFrom1Subj); % Track size of collection
end

% Removed unused space
features((totalDataAdded+1):end,:) = [];
labels((totalDataAdded+1):end) = [];
unModLabels((totalDataAdded+1):end) = [];
subjectList(isnan(subjectList)) = [];
recordNames = recordNames(~cellfun(@isempty,recordNames));

if (length(features) ~= length(labels) || length(features) ~= length(unModLabels))
    error('Missing data or annotations');
end

disp(['Removed ' num2str(100*sum(artifactRegion)/length(labels)) '%% of features'])
end

function [inputFeatures, oneHotLabels] = prepareInputsForTraining(features, labels, stages)
% Transform labels into a output vector usable for training the perceptron
% as well as eliminate samples that will not be used in training
% ('undefined, 'n1' and 'n2')
[numFeatureVecs,featuresPerVector] = size(features);
oneHotLabels = zeros(numFeatureVecs,4);
inputFeatures = zeros(numFeatureVecs,featuresPerVector);
parfor i = 1:numFeatureVecs
    % Assign output vector of binaries to each of the stages used
    if labels(i) == stages('Wake')
        oneHotLabels(i,:) = [1 0 0 0];
        inputFeatures(i,:) = features(i,:);
    elseif labels(i) == stages('REM')
        oneHotLabels(i,:) = [0 1 0 0];
        inputFeatures(i,:) = features(i,:);
    elseif labels(i) == stages('Light')
        oneHotLabels(i,:) = [0 0 1 0];
        inputFeatures(i,:) = features(i,:);
    elseif labels(i) == stages('Deep')
        oneHotLabels(i,:) = [0 0 0 1];
        inputFeatures(i,:) = features(i,:);
    end
    % Otherwise, ignore it
end
end

function [labels, arousals] = getLabelsFromAnn(fileName,annName,recordFs,signalSize,translatorMap,stages,workingDir,runName)
% Obtain labels from annotation file
[~, dataStr] = system(['rdann -r ' fileName ' -a ' annName]);
dataRows = strsplit(dataStr,'\n'); % Split data into rows
L = length(dataRows)-1;

labels = nan(signalSize,1); % Leave as nan for regions in which sleep stage was not scored
arousals = zeros(signalSize,1);
lastStageEnd = nan;

for i = 1:L
    % Loop through rows of data matrix
    splitRow = strsplit(dataRows{i},{' ','\t'}); % Get items in single row
    index = str2num(splitRow{3}); % Get index of 1st sample in event
    dur = str2num(splitRow{9}); % Duration in seconds of event
    if strcmp(splitRow{8}(1:6),'SLEEP-')
        % If event is sleep stage, get label of stage and apply numerical
        % label to all samples in array
        if dur ~= 30
            warning(['Duration not 30 for annotation row ' num2str(i)])
        end
        assert(isnan(lastStageEnd) || (index >= lastStageEnd),'data_extraction:temporally_inconsistent_indices','Indices suggest overlapping sleep stages.')
        lastStageEnd = index+dur*recordFs; % Track index where previous sleep stage ended for use in error detection
        stage = splitRow{10};
        if any(strcmp(keys(translatorMap),stage))
            labels((index+1):(index+dur*recordFs)) = repelem(stages(translatorMap(stage)),dur*recordFs);
        else
            qdisp(['Could not find stage ' stage '. Leaving undefined.'],workingDir,runName)
            labels((index+1):(index+dur*recordFs)) = repelem(stages('undefined'),dur*recordFs);
        end
    else
        % If event is arousal, indicate all samples in array during which
        % arousal is occuring
        arousals((index+1):(index+dur*recordFs)) = repelem(1,dur*recordFs);
    end
end
end

function labels = extractLabels(annotations,stages)
% Obtain labels from sleep_stages struct

% Check to make sure no overlap
if any((annotations.wake + annotations.rem + annotations.nonrem1 + annotations.nonrem2 + annotations.nonrem3 + annotations.undefined) > 1)
    error('Error: section of data labeled as multiple sleep stages simultaneously');
end

% Create list of numerical labels fore each sleep stage.
labels = stages('awake')*annotations.wake ...
    + stages('n1')*annotations.nonrem1 ...
    + stages('n2')*annotations.nonrem2 ...
    + stages('n3')*annotations.nonrem3 ...
    + stages('rem')*annotations.rem ...
    + stages('undefined')*annotations.undefined;
end

function normalizedFeatures = getEEGAutoregressFeatures(data,Fs,ARorder)
% Train Pardey's model on data at sampling frequecy FS
blockSize = 1; % Size of block in seconds to partition data into
L = floor(length(data)/(Fs*blockSize));
features = zeros(L,ARorder);

data = (data - mean(data))/std(data);
parfor i = 1:L
    % Get autoregression or each block using Burg's method
    [~,refl] = ar(data((((i-1)*blockSize*Fs)+1):(i*blockSize*Fs)),ARorder,'burg'); % Obtain reflection coefficients
    features(i,:) = refl(1,2:end); % Omit the first term, which is always 0
end

% 0-mean, unit-variance normalization of features (Note: I am not
% entirely sure if this is exactly the way the original paper did it)
normalizedFeatures = zeros(length(features),ARorder);
for i = 1:ARorder
    normalizedFeatures(:,i) = (features(:,i) - nanmean((features(:,i))))/nanstd(features(:,i));
end
end

function [features, labels, unModLabels, subjectList] = removeUnwantedFeatures(features, labels, unModLabels, subjectList, removeTransitions, removeArousals, sleepStages)
% Removes samples marked as 'undefined' or NaN, and removes epochs that are
% on the boarder between sleep stages, if specified.
epochLength = 30;
keepAndRemove = false(1,length(labels)); % Boolean array of which points to keep or remove

if (removeTransitions)
    % Remove data before and after sleep stage switches from one to another
    transitionPoints = find(diff(unModLabels) ~= 0); % Find points where sleep stage changes
    for i = 1:length(transitionPoints)
        % Mark regions 30s before and after transition points for removal
        keepAndRemove((transitionPoints(i)-29):(transitionPoints(i)+30)) = true;
    end
end

if (removeArousals)
    % Remove arousal regions (ie: regions where region was reclassified as
    % wake)
    for i = 1:(length(labels)/epochLength)
        if any(labels(((i-1)*epochLength+1):(i*epochLength)) ~= unModLabels(((i-1)*epochLength+1):(i*epochLength)))
            keepAndRemove(((i-1)*epochLength+1):(i*epochLength)) = true;
        end
    end
end

keepAndRemove(unModLabels == sleepStages('undefined')) = true; % Mark undefined features for removal
%keepAndRemove(unModLabels == sleepStages('n1')) = true; % Mark stage 1 features for removal
subjectList = adjustForRemovedSamples(subjectList,keepAndRemove); % Modify indices of where data from each subject begins to account for feature removal

% Remove marked features
features(keepAndRemove, :) = [];
unModLabels(keepAndRemove) = [];
labels(keepAndRemove) = [];

disp(['Eliminated ' num2str(100*sum(keepAndRemove)/length(keepAndRemove)) '%% of features'])

assert(~any(isnan(features),'all'),'Not all NaN''s removed from features');
assert(length(labels) == length(features),'Labels and features are not the same size.');
end

function [features,labels,unModLabels,newSubjectList,recordNames] = removeOutOfAgeRange(features,labels,unModLabels,subjectList,ageData,minAge,maxAge,accountForAge,recordNames)
shortAgeData = ageData(1:length(subjectList)); % Remove unused subjects from list of ages if not using all subjects
wantedSubjectIndices = shortAgeData <= maxAge & shortAgeData >= minAge;
if ~isempty(recordNames)
    % If not empty, remove recordNames of removed subjects
    recordNames(~wantedSubjectIndices) = [];
end
wantedSubjectList = find(wantedSubjectIndices); % Get list of subjects within age range
L = length(wantedSubjectList);
newSubjectList = zeros(1,length(wantedSubjectList));
dataToKeep = zeros(1,length(labels));
ageVector = zeros(length(labels),1);
for i = 1:L
    if (wantedSubjectList(i) < length(subjectList))
        newSubjectList(i) = sum(dataToKeep) + 1; % keep list of where data of each subject starts and ends
        dataToKeep(subjectList(wantedSubjectList(i)):(subjectList(wantedSubjectList(i)+1)-1)) = 1; % get indices of data belonging to subjects within age range
        ageVector(subjectList(wantedSubjectList(i)):(subjectList(wantedSubjectList(i)+1)-1)) = shortAgeData(wantedSubjectList(i)); % get age of subject for each sample used
    else
        % If going to keep data from last subject in subject list, just mark all remaining data as 'keep'
        newSubjectList(i) = sum(dataToKeep) + 1; % keep list of where data of each subject starts and ends
        dataToKeep(subjectList(wantedSubjectList(i)):end) = 1; % get indices of data belonging to subjects within age range
        ageVector(subjectList(wantedSubjectList(i)):end) = shortAgeData(wantedSubjectList(i)); % get age of subject for each sample used
    end
end

% Remove unwanted data
subjectList = adjustForRemovedSamples(subjectList,~dataToKeep);
features(find(~dataToKeep),:) = [];
labels(find(~dataToKeep)) = [];
unModLabels(find(~dataToKeep)) = [];
ageVector(find(~dataToKeep)) = [];


if accountForAge
    % If indicated to do so, add age as a feature
    features = [features ageVector];
end
end

function [adjustedSubjectList] = adjustForRemovedSamples(originalSubjectList,removedSamples)
% Adjust indices in 'subjectList' to account for removal of samples specified
% in logical array 'removedSamples'
L = length(originalSubjectList);
adjustedSubjectList = zeros(1,L);
adjustedSubjectList(1) = 1; % Know first sample will be at index 1
for i = 2:L
    % Find total removed samples up to each index and subtract from
    % original index to get adjusted index
    adjustedSubjectList(i) = originalSubjectList(i) - sum(removedSamples(1:(originalSubjectList(i)-1)));
end
assert(~any(adjustedSubjectList == 0),'Error adjusting subject list for removed samples - no subject data should start at ''0''')

adjustedSubjectList(diff(adjustedSubjectList)==0) = []; % Remove repeating elements to eliminate subjects whose entire dataset was removed
if all(removedSamples(originalSubjectList(end):end))
    % Detect if all samples from last subject were removed (as this would
    % not produce duplicate elements) and remove index for the last subject
    % if this is the case
    adjustedSubjectList(end) = [];
end
end

function [features,labels,unModLabels,subjectList,recordNames] = removeIgnoredDz(features,labels,unModLabels,subjectList,recordNames,ignoreDz)
% Remove features from subjects with diseases to ignore
diseaseNames = cellfun(@(name)lower(name(1:find(isstrprop(name,'digit'),1))),recordNames,'UniformOutput',false); % Remove numbers from each record name, leaving only disease name
L = length(recordNames)-1;
for i = 1:L
    if (any(strcmp(diseaseNames{i},ignoreDz)))
        % Mark data belonging to patients with unwanted diseases for later
        % removal
        labels(subjectList(i):(subjectList(i+1)-1)) = nan;
        recordNames{i} = {};
    end
end

% Repeat above step slightly modified for the final subject in the list
if (any(strcmp(diseaseNames{end},ignoreDz)))
    % Mark data belonging to patients with unwanted diseases for later
    % removal
    labels(subjectList(end):end) = nan;
    recordNames{end} = {};
end


% Use values in 'labels' marked 'nan' to indicate number and location of
% samples were removed to adjust indices of subject data in subjectList
subjectList = adjustForRemovedSamples(subjectList,isnan(labels));

% Remove values marked as unwanted
features(isnan(labels),:) = []; % Use indices of labels marked nan to remove feature vectors
labels(isnan(labels)) = [];
unModLabels(isnan(labels)) = [];
recordNames = recordNames(~cellfun('isempty',recordNames));

assert(length(recordNames) == length(subjectList),'Subjects or recordNames incorrectly removed')
end

function [features,labels,unModLabels,subjectList] = balanceClassesByDz(features, labels, unModLabels, subjectList, recordNames)
% Remove samples so that each class has the same number of samples from
% each disease

dzNames = cellfun(@(name)lower(name(1:find(isstrprop(name,'digit'),1))),recordNames,'UniformOutput',false); % Remove numbers from each record name, leaving only disease name
dzTypes = unique(dzNames); % Get list of all unique disease types
numDzNames = length(dzNames);
numDzTypes = length(dzTypes);
samplesPerDzType = zeros(1,numDzTypes);
rng(7); % For replicability

% Tally total members of each disease type
for i = 1:numDzTypes
    % Loop through disease types
    for j = 1:numDzNames
        % Loop through list of subjects
        if strcmp(dzNames{j},dzTypes{i})
            if (j < numDzNames)
                samplesPerDzType(i) = samplesPerDzType(i) + length(subjectList(j):(subjectList(j+1)-1)); % Tally samples from each subject with specified disease
            else
                % Same as above line, but modified when subject is last in
                % list
                samplesPerDzType(i) = samplesPerDzType(i) + length(subjectList(end):length(labels)); % Tally samples from each subject with specified disease
            end
        end
    end
end

% Reduce sizes of all classes so that they're all the same length (ignore
% empty classes)
samplesToKeep = min(samplesPerDzType(samplesPerDzType > 0)); % Find number of samples to reduce to

% Randomly remove samples from subjects such that all diseases are equally
% represented
for i = 1:numDzTypes
    % Loop through disease types
    for j = 1:numDzNames
        % Loop through list of subjects
        if strcmp(dzNames{j},dzTypes{i})
            % Determine number of features necessary to remove from each
            % disease type, spread quantity to remove evenly across
            % subjects, then mark random features for removal
            if (j < numDzNames)
                % Determine range of samples belonging to subject
                featureRange = subjectList(j):(subjectList(j+1)-1);
            else
                % Same as above line, but modified when subject is last in
                % list
                featureRange = subjectList(j):length(labels); % Determine range of samples belonging to subject
            end
            
            numWithDz = sum(find(strcmp(dzTypes{i},dzNames))); % Find total number of subjects with disease
            numFeaturesToRemove = round((samplesPerDzType(i) - samplesToKeep)/numWithDz); % Determine number of features to remove from each subject
            featuresToRemove = featureRange(randi(length(featureRange),numFeaturesToRemove)); % Randomly select features to remove
            labels(featuresToRemove) = nan;
        end
    end
end

% Use values in 'labels' marked 'nan' to indicate number and location of
% samples were removed to adjust indices of subject data in subjectList
subjectList = adjustForRemovedSamples(subjectList,isnan(labels));

% Remove values marked as unwanted
features(isnan(labels),:) = [];
unModLabels(isnan(labels)) = [];
labels(isnan(labels)) = [];
end

function newStartCell = moveCell(startCell,addRows,addCols)
% Generate possible excel column names
alphabet = 'A':'Z';
colNames = cell(1,26*(26+1));
colNames(1:26) = num2cell(alphabet);
% Generate list of possible excel column names
for i = 1:length(alphabet)
    colNames((i*26+1):((i+1)*26)) = strcat(alphabet(i),num2cell(alphabet));
end

% Modify the location of 'startCell'
startCol = startCell(isletter(startCell)); % Get column of first cell to write to
colNum = find(strcmp(startCol,colNames));
startRow = str2num(startCell(~isletter(startCell))); % Get row number of first cell to write to

% Adjust column and row numbers
newColNumber = colNum + addCols;
newRowNumber = startRow + addRows;

newStartCell = [colNames{newColNumber} num2str(newRowNumber)]; % Create name of new startCell
end

function array2libsvm(features,labels,fname)
% Convert feature array 'features' and corresponding labels array 'labels'
% into format readable by libSVM/thunderSVM and write to file 'fname'
%
% If a value in the feature set is missing, set it to NaN so that it will
% be ignored.

[numRows,numCols] = size(features);
fid = fopen(fname, 'w');
for i = 1:numRows
    fprintf(fid,'%g',labels(i));
    for j = 1:numCols
        if ~isnan(features(i,j))
            fprintf(fid,' %g:%g',j,features(i,j));
        end
    end
    fprintf(fid,'\n');
end
fclose(fid);
end

function thunderSvmTrain(params,features,labels,modelName,workingDir)
% Train thunderSVM on dataset.
% params - SVM parameters
% features - feature matrix
% labels - label array
% modelName - name of file model shoud be saved to

dataTmpFileName = [workingDir 'tmp_' num2str(randi(1e6,1,1)) '.txt'];
array2libsvm(features,labels,dataTmpFileName); % Store data in file readable to thunderSVM

% Detect if gpu is available
try
    gpuArray(1);
    canUseGPU=true;
catch
    canUseGPU=false;
end

% Train model
if canUseGPU
    % Use the appropriate executable depending on whether gpu available
    system([workingDir 'gpuSvm/thundersvm-train ' params ' "' dataTmpFileName '" "' modelName '"'])
else
    pool = gcp('nocreate');
    system([workingDir 'noGpuSvm/thundersvm-train ' params ' -o ' num2str(pool.NumWorkers) ' "' dataTmpFileName '" "' modelName '"'])
end
delete(dataTmpFileName); % Remove tmp files
end

function labels = thunderSvmPredict(features,modelName,workingDir)
% Train thunderSVM on dataset.
% params - SVM parameters
% features - feature matrix
% labels - label array
% modelName - name of file model should be saved to

dataTmpFileName = [workingDir 'tmp_' num2str(randi(1e6,1,1)) '.txt'];
outputTmpFileName = [workingDir 'tmp_' num2str(randi(1e6,1,1)) '.predict'];
array2libsvm(features,zeros(size(features,1),1),dataTmpFileName); % Write features to file usable by thunderSVM (labels unimportant, so set all to 0)

% Detect if gpu is available
try
    gpuArray(1);
    canUseGPU=true;
catch
    canUseGPU=false;
end

% Label features
[~,tester]=system('ls');
disp(tester)
if canUseGPU
    % Use the appropriate executable depending on whether gpu available
    system([workingDir 'gpuSvm/thundersvm-predict "' dataTmpFileName '" "' modelName '" "' outputTmpFileName '"'])
else
    system([workingDir 'noGpuSvm/thundersvm-predict "' dataTmpFileName '" "' modelName '" "' outputTmpFileName '"'])
end
labels = dlmread(outputTmpFileName);

delete(dataTmpFileName); % Remove tmp files
delete(outputTmpFileName)
end

function kappaPerSubj = getKappaPerSubj(features,labels,subjectList,predictor)

% Cycle through list of subjects and evaluate cohen's Kappa of perceptron predictions
L = length(subjectList);
subjectList(end+1) = length(labels)+1;
kappaPerSubj = zeros(L,1);
for i = 1:L
    % Get beginning and ending indices for data of patient
    start = subjectList(i);
    stop = subjectList(i+1)-1;
    
    % Get data from each patient
    featuresFrom1Subj = features(start:stop,:);
    labelsFrom1Subj = labels(start:stop);
    
    % Predict sleep stage
    predictions = predictor(featuresFrom1Subj); % Predict sleep category
    
    % Evaluate accuracy
    kappaPerSubj(i) = cohenKappa(predictions,labelsFrom1Subj);
end
end

function [features, labels] = balanceClassesBySample(features, labels, stages)
% Remove samples so that each class has the same number of samples
stageList = values(stages);
numStages = length(stageList);
membersPerClass = zeros(1,numStages);
markForRemoval = false(1,length(labels));

% Tally total members of each class
for i = 1:numStages
    membersPerClass(i) = sum(labels == stageList{i});
end

% Reduce sizes of all classes so that they're all the same length (ignore
% empty classes)
numSamplesToKeep = min(membersPerClass(membersPerClass > 0));
for i = 1:numStages
    if membersPerClass(i) ~= 0
        reduceBy = membersPerClass(i) - numSamplesToKeep; % Number of samples of class to eliminate
        markForRemoval(datasample(find(labels == stageList{i}),reduceBy,'Replace',false)) = true; % Randomly select samples from each class to eliminate
    end
end

% Remove marked samples
features(markForRemoval,:) = [];
labels(markForRemoval) = [];

if(size(features,1) ~= length(labels))
    error('Features or labels missing after balancing.')
end
end


function [features, labels] = balanceClassesByEpoch(features, labels, stages)
% Remove samples an entire epoch at a time so that each class has the same number of samples
stageList = values(stages);
numStages = length(stageList);
membersPerClass = zeros(1,numStages);
epochLength = 30;

% Tally total members of each class
for i = 1:numStages
    membersPerClass(i) = sum(labels == stageList{i});
end
assert(sum(mod(membersPerClass,epochLength)) == 0,'Numbers of each class should be integer multiple of epoch length');

% Reduce sizes of all classes so that they're all the same length
numSamplesToKeep = min(membersPerClass(membersPerClass ~= 0));
epochLabels = labels(1:epochLength:end); % Get label of samples in each epoch
epochForRemoval = false(1,length(labels)/epochLength); % Mark which epochs will be removed
for i = 1:numStages
    if membersPerClass(i) ~= 0
        reduceBy = (membersPerClass(i) - numSamplesToKeep)/epochLength; % Number of epochs of class to eliminate
        epochForRemoval(datasample(find(epochLabels == stageList{i}),reduceBy,'Replace',false)) = true; % Randomly select samples from each class to eliminate
    end
end
samplesForRemoval = repelem(epochForRemoval,epochLength);

% Remove marked samples
features(samplesForRemoval,:) = [];
labels(samplesForRemoval) = [];

% Tally total members of each class again to check success
for i = 1:numStages
    membersPerClass(i) = sum(labels == stageList{i});
end
assert(all(membersPerClass(membersPerClass ~=0 ) == membersPerClass(1)),'Features not correctly balanced')
assert(sum(mod(sum(labels == unique(labels)'),epochLength)) == 0,'Numbers of each class should be integer multiple of epoch length')
assert(size(features,1) == length(labels),'Features or labels missing after balancing')
end

function [oversampledFeatures,oversampledLabels] = balanceByUpsampling(features,labels,stages,upsamplingFunc,categoriesToFix,varargin)
% Take matrix of features 'features', corresponding labels 'labels', list
% of subjects subjectList and the numerical labels of the classes to fix
% 'categoriesToFix' and returns the features and labels of the classes after
% they've been upsampled via SMOTE, along with the adjusted list of subjects.
assert(~isempty(categoriesToFix),'Need at least 1 sleep category fixed')

stageLabelList = cell2mat(values(stages));
numStages = length(stageLabelList);
membersPerClass = zeros(1,numStages);
epochLength = 30;

% Tally total members of each class
for i = 1:numStages
    membersPerClass(i) = sum(labels == stageLabelList(i));
end
assert(sum(mod(membersPerClass,epochLength)) == 0,'Numbers of each class should be integer multiple of epoch length');

% Determine how many of the non-fixed classes need to be generated
upsampleTo = max(membersPerClass);
assert(all(upsampleTo >= membersPerClass),'Combined size of fixed classes must be at least equal to size of all other classes')

% Create new array with space for synthetic samples
totalAdditionalSamples = sum(upsampleTo - membersPerClass(membersPerClass ~=0 & ~ismember(stageLabelList,categoriesToFix)));
oversampledFeatures = [features; nan(totalAdditionalSamples,size(features,2))];
oversampledLabels = [labels; nan(totalAdditionalSamples,1)];
totalDataAdded = length(labels);

% Cycle through each non-fixed & non-empty class
for i = 1:numStages
    if (membersPerClass(i) ~= 0 && ~ismember(stageLabelList(i),categoriesToFix))
        % Generate samples via smote
        numToGenerate = upsampleTo - membersPerClass(i);
        oversampledFeatures((totalDataAdded+1):totalDataAdded+numToGenerate,:) = upsamplingFunc(features(labels == stageLabelList(i),:),numToGenerate,varargin{:});
        oversampledLabels((totalDataAdded+1):totalDataAdded+numToGenerate) = stageLabelList(i);
        totalDataAdded = totalDataAdded + numToGenerate;
    end
end
assert(~any(isnan(oversampledFeatures),'all'),'Synthetic features generated incorrectly')
end

function [balancedFeatures, balancedLabels, balancedSubjectList] = balanceBySelectedMethod(features,labels,stages,subjectList,method,varargin)
% Balance classes by patient using the selected upsampling or subsampling method

switch method
    % Choose method used to balance classes
    case 'smote'
        selectedBalanceMethod = @(features,labels) balanceByUpsampling(features,labels,stages,@smoteUpsample,varargin{:});
    case 'nnetSmote'
        selectedBalanceMethod = @(features,labels) balanceByUpsampling(features,labels,stages,@nnetUpsample,varargin{:});
    case 'subsampleEpochs'
        selectedBalanceMethod = @(features,labels) balanceClassesByEpoch(features, labels,stages);
    case 'subsampleSamples'
        selectedBalanceMethod = @(features,labels) balanceClassesBySample(features, labels,stages);
    case 'gmm'
        selectedBalanceMethod = @(features,labels) balanceByUpsampling(features, labels,stages,@gmmSynthSamps,varargin{:});
end

L = length(subjectList);
subjectList(end + 1) = length(labels)+1;
balancedFeatures = nan(10*3600*length(subjectList),size(features,2)); % Initialize arrays to larger than needed and shrink later
balancedLabels = nan(10*3600*length(subjectList),1);
balancedSubjectList = [];
totalFeaturesAdded = 1;

for i = 1:L
    % Get data from individual patients
    featuresFrom1Subj = features(subjectList(i):(subjectList(i+1)-1),:);
    labelsFrom1Subj = labels(subjectList(i):(subjectList(i+1)-1));
    
    if ~all(sum(unique(labelsFrom1Subj) == cell2mat(values(stages)),1))
        % Skip patients who don't have any of a particular feature
        continue
    end
    
    % Balance by individual patient
    [balancedFeaturesFrom1Subj,balancedLabelsFrom1Subj] = selectedBalanceMethod(featuresFrom1Subj,labelsFrom1Subj);
    
    % Add data to larger collection
    balancedSubjectList(end+1) = totalFeaturesAdded;
    balancedFeatures(totalFeaturesAdded:(totalFeaturesAdded + size(balancedLabelsFrom1Subj,1) - 1),:) = balancedFeaturesFrom1Subj;
    balancedLabels(totalFeaturesAdded:(totalFeaturesAdded + length(balancedLabelsFrom1Subj) - 1),:) = balancedLabelsFrom1Subj;
    totalFeaturesAdded = totalFeaturesAdded + length(balancedLabelsFrom1Subj);
end

balancedFeatures(totalFeaturesAdded:end,:) = [];
balancedLabels(totalFeaturesAdded:end,:) = [];

assert(~any(isnan(balancedFeatures),'all'),'Features incorrectly collected during class-balancing')
assert(~any(isnan(balancedLabels),'all'),'Labels incorrectly collected during class-balancing')
end

function [features, labels, unModLabels, subjectList] = removeUnwantedSubjects(features, labels, unModLabels, subjectList, removeSubjects)
% Removes all data from patients in removeSubjects

L = length(subjectList);
keepAndRemove = false(1,length(labels)); % Boolean array of which points to keep or remove

subjectList(end+1) = length(labels)+1; % Append number to end of patient list marking index of last sample
for i = 1:L
    if ismember(i,removeSubjects)
        % If subject is in list of subjects to remove, mark all data from
        % subject for removal
        keepAndRemove(subjectList(i):(subjectList(i+1)-1)) = true;
    end
end
subjectList(end) = []; % Remove appended number


subjectList = adjustForRemovedSamples(subjectList,keepAndRemove); % Modify indices of where data from each subject begins to account for feature removal

% Remove marked features
features(keepAndRemove, :) = [];
unModLabels(keepAndRemove) = [];
labels(keepAndRemove) = [];

assert(~any(isnan(features),'all'),'Not all NaN''s removed from features');
assert(length(labels) == length(features),'Labels and features are not the same size.');
end

function [sol,tst,waso,se] = calcSleepStats(labels,stages,startIndList,epochLength)
% Find various sleep statistics of all subjects in list
% 
% INPUTS
% labels -------- Array of sleep stage labels for each epoch
% stages -------- Map linking sleep stage to numerical labels used in 'labels'
% startIndList -- Locations of first element in 'labels' where each new subject's data starts
% epochLength --- Duration (s) of each epoch
% 
% OUTPUTS
% sol ----------- Sleep onset latency
% tst ----------- Total sleep time
% waso ---------- Wake after sleep onset
% se ------------ Sleep efficiency

L = length(startIndList);
sol = nan(1,L);
tst = nan(1,L);
waso = nan(1,L);
se = nan(1,L);

startIndList(end+1) = length(labels);
for i = 1:L
    subjData = labels(startIndList(i):startIndList(i+1)); % Get all data from 1 subject
    sol(i) = epochLength*find(subjData ~= stages('Wake'),1); % Sleep onset latency is time of first sample not equal to 'Wake' 
    tst(i) = epochLength*sum(subjData ~= stages('Wake'));
    waso(i) = epochLength*length(subjData) - tst(i) - sol(i);
    se(i) = 100*tst(i)/(epochLength*length(subjData));
end

% Sanity checking
assert(all(~isnan(sol)),'Sleep onset latencies calculated incorrectly')
assert(all(~isnan(tst)),'Total sleep time calculated incorrectly')
assert(all(~isnan(waso)),'Wake after sleep onset calculated incorrectly')
assert(all(~isnan(se)),'Sleep efficiency calculated incorrectly')
end

