# SingleChanSleepSimple
Low complexity single channel sleep stager

## Main Function 
**predictor = trainPardeyMaster(trainingRecords,channel,workingDir,varargin)**

Master code for training and testing sleep-staging algorithm on a set of physionet records.
Currently supports formatting conventions used by Physionet's 2018 Challenge, Cyclic Alternating
Pattern and Sleep Heart Health Study datasets.
Performs 5-fold cross-validation unless specified to do otherwise using the -noCrossVal switch or when
the validation set is specified.

In the output directory, the following files will be created:

\<_DATE_\>\_predictor.mat - Contains trained model. Can be reloaded and tested again using the 'savedMdls' optional argument or used to create ensemble model.\
\<_DATE_\>\_kappaPerSubj\<_FOLD NUMBER_\>.mat - Contains list of kappa value of trained model prediction accuracy for each subject\
\<_DATE_\>\_train_confusion.fig - Confusion plot of the trained model evaluated on the training data\
\<_DATE_\>\_validation_confusion.fig - Confusion plot of the trained model evaluated on the validation data
  
**REQUIRED INPUTS**
Variable | Description
--- | --- 
trainingRecords | (_string_) Comma-separated list of .mat or .edf files to extract EEG data from (ex: 'file1.mat,file2.mat').
channel | (_string or cell array of strings_) Name of channel (ex: 'c4a1'), or possible names of channels if same channel has multiple possible names (ex: {'c4a1','C4-A1'})
workingDir | (_string_) Directory to save outputs or temporary files to.
 
**OUTPUTS**
Variable | Description
--- | --- 
predictor | (_ComboNet object_) Trained model used for predicting sleep stages 
 
**OPTIONAL FLAGS**
Flag | Description
--- | --- 
-noCrossVal | If this switch is present, perform single run instead of 5-fold cross-validation. If validation set not specified, use the 0th fold.
-removeArousals | If this switch is present, remove epochs containing arousal events
-noFeatureRemoval | If this switch is present, avoid removing dropped-out features during feature extraction.
-removeTrainTrans | If this flag is present, remove 30s of data at both sides of the points at which the subject transitions from one sleep stage to another in the training data
-removeValTrans | If this flag is present, remove 30s of data at both sides of the points at which the subject transitions from one sleep stage to another in the validation data

**OPTIONAL NAME-VALUE PAIRS**
Name | Value Description
--- | --- 
nnetNodes | (_int array_) Array of widths of each layer in network. Only used if creating new weak learner. Default = [6]
ensemble | (_string_) If provided, train ensemble model of provided weak learners instead of single new neural network. Value should be name of .mat file containing cell array of ComboNet objects named
savedFeatures | (_string_) .mat containing extracted features, labels, and list of indices in the feature/label vectors where each patient's data begins. This .mat file is created whenever extracting features and can be re-loaded whenever on later runs in order to skip the feature extraction step.
savedMdls | (string) Name of .mat file containing saved ComboNet object. If present, load the saved ComboNet object and evaluate on validation set without training.
ARorder | (_int_) Autoregression order used for extracting reflection coefficients with Burg's algorithm. Not used if 'savedFeatures' value is given. Unused if loading saved features. Default = 10.
validationSetChosen | (_int array_) Specify exactly which subjects to include in validation set as indices in the list of subjects (for example, if using 5 subjects total, passing the argument [1,3,4] would cause the first, third and fourth subjects to be used for validation while the rest are used for training). 
removeSubjects | (_int array_) Specify subjects not to include in training or test.
writeReport | (_string_) Specify location where confusion matrix and performance can be printed do in format '\<_FILE NAME_\>,\<_SHEET NAME_\>,\<_START CELL_\>'.
labelFiles | (_string_) If labels in separate directory from data files, specify label file location and extension as two separate values (ex: trainPardeyMaster(...,'labelFiles','path/labelDir',',txt')). It is that each data file has a corresponding label file with the same name
Fs | (_numeric_) Sampling frequency of EEG in records. Unused if loading saved features.
runName | (_string_) Providing this value will cause any file that is saved to be saved into a directory within _workingDir_ with this name.
notationTranslator | (_string_) Map sleep staging labels from any database to the labels used in the cinc 2018 dataset. Example: to map the labels used in the sleep heart health study dataset (W,REM or R,S4,S3,S2,S1) to the notation used in Cinc 2018 (awake,rem,n3,n2,n1), the following string is used: 'W=awake,REM=rem,R=rem,S4=n3,S3=n3,S2=n2,S1=n1'. Notice that multiple labels from one dataset can be translated into one label used by Cinc, for example, when the labels 'REM' and 'R' are both used interchangeably to refer to REM sleep. It can also be used to deal with datasets which use R&K instead of AASM rules by mapping both stage 4 and stage 3 to the 'n3' label.
