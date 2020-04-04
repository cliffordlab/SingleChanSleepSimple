classdef ComboNet
    
    properties
        learners = {} % List of weak learners
        weights = [];
    end
    properties (Dependent)
        layers
    end
    properties (SetAccess = private,GetAccess = public)
        type = '' % Type of combination
    end
    properties (Constant = true)
        epochLength = 30;
    end
    methods
        function comboNet = ComboNet(type,learners,varargin)
            comboNet.type = type;
            comboNet.learners = learners;
            
            % Error checking
            assert(all(comboNet.learners{1}.layers{end}.size == cell2mat(cellfun(@(x) x.layers{end}.size,comboNet.learners,'UniformOutput', false))), ...
                'All neural networks should have same number of outputs')
            assert(all(comboNet.learners{1}.layers{1}.size == cell2mat(cellfun(@(x) x.layers{1}.size,comboNet.learners,'UniformOutput', false))), ...
                'All neural networks should have same number of inputs')
        end
        
        function layers = get.layers(self)
            assert(strcmp(self.type,'singleton'),'Property ''layers'' only defined for ComboNet objects of type ''singleton''')
            layers = self.learners{end}.layers;
        end
        
        function self = train(self,features,oneHotLabels)
            switch self.type
                case 'singleton'
                    % Use single perceptron
                    trainedPerceptron = singletonTrain(self,features', oneHotLabels');
                    self.learners = {trainedPerceptron};
                case 'softVote'
                    % No training needed
                case 'weightedSoftVote'
                    self = regressWeights(self,@weightedSoftVote,features',oneHotLabels');
                case 'weightedHardVote'
                    self = regressWeights(self,@weightedHardVote,features',oneHotLabels');
                case 'mostConfident'
                    % No training needed
                case 'stack'
                    % TODO
            end
            disp('Model Trained. Weights = ')
            disp(self.weights)
        end
        
        function self = regressWeights(self,predictFunc,features,oneHotLabels)
            % Find weights which optimizes prediction accuracy of
            % predictFunc
            N = length(self.learners);
            regressFunc = @(weights,features) predictFunc(self,features,weights);
            self.weights = lsqcurvefit(regressFunc, ...
                ones(N,1)/N, ...
                features, ...
                vec2ind(oneHotLabels)', ...
                [], ...
                [], ...
                optimoptions('lsqcurvefit','UseParallel',true,'FiniteDifferenceStepSize',.5,'Algorithm','levenberg-marquardt')); % Perform regression
        end
        
        function predictions = subsref(self,S)
            % Allows comboNet predict function to be called in same method
            % as network 'sim' function (ie: prediction = comboNet(features))
            assert(strcmp(S.type,'()'),'Use parentheses to call obtain predictions (ex: out = comboNet(input))')
            predictions = predict(self,S.subs{1});
        end
        
        function predictions = predict(self,features)
            switch self.type
                case 'singleton'
                    predictions = singletonPredict(self,features');
                case 'softVote'
                    predictions = softVote(self,features');
                case 'weightedSoftVote'
                    predictions = weightedSoftVote(self,features',self.weights);
                case 'mostConfident'
                    predictions = mostConfident(self,features');
                case 'weightedHardVote'
                    predictions = weightedHardVote(self,features',self.weights);
                case 'stack'
                    % TODO
            end
        end
        
        function predictions = singletonPredict(self,features)
            % Predict using single perceptron output
            scores = self.learners{1}(features)'; % Obtain output of perceptron
            [~,predictions] = max(scores,[],2); % Get index (and therefore label) of highest values in each row
            
            predictions = epochVote(self,predictions);
        end
        
        function trainedNet = singletonTrain(self,inVecs, targetVecs)
            % Train neural network
            trainedNet = train(self.learners{1},inVecs,targetVecs,'useParallel','yes'); % 'useGPU','yes'
        end
        
        function predictions = softVote(self,features)
            % Weighted soft vote with all weights same
            N = length(self.learners);
            predictions = weightedSoftVote(self,features,ones(1,N)/N); % All weights same
        end
        
        function predictions = weightedSoftVote(self,features,weights)
            % Take weighted average of each neural network and pick class
            % with highest probability
            N = length(self.learners);
            L = size(features,2);
            aveOut = zeros(self.learners{1}.layers{end}.size,L);
            for i = 1:N
                singleOut = self.learners{i}.learners{1}(features); % Get output of each network
                aveOut = aveOut + weights(i)*singleOut; % Take weighted average of each output
            end
            
            [~,predictions] = max(aveOut,[],1); % Get index (and therefore label) of highest values in each row
            
            predictions = epochVote(self,predictions)';
        end
        
        
        function predictions = weightedHardVote(self,features,weights)
            % Weighted vote of prediction of each learner
            dim = self.learners{1}.layers{end}.size;
            N = length(self.learners);
            L = size(features,2);
            voteWeights = zeros(dim,L);
            
            for i = 1:N
                singleOut = self.learners{i}.learners{1}(features); % Get output of each network
                [~,singlePrediction] = max(singleOut,[],1); % Vote goes to class with highest probability
                voteWeights = voteWeights + weights(i)*ind2vec(singlePrediction,dim); % Weighted sum of each vote at each sample
            end
            
            [~,predictions] = max(voteWeights,[],1); % Label goes to class with most weight
            predictions = epochVote(self,predictions)';
        end
        
        function predictions = mostConfident(self,features)
            % Pick label of most confident learner (ie: learner where most probable
            % label is furthest from other labels) for each sample
            m = self.learners{1}.layers{end}.size;
            N = length(self.learners);
            L = length(features);
            singlePredictions = zeros(N,L);
            confidence = zeros(size(singlePredictions));
            
            for i = 1:N
                nnetOuts = self.learners{i}.learners{1}(features); % Get output of each network
                [~,singlePredictions(i,:)] = max(nnetOuts,[],1); % Get class predicted by each network
                for j = 1:L
                    % Confidence is average difference between probability
                    % of most probable class and probability of other classes
                    confidence(i,j) = mean(nnetOuts(singlePredictions(i,j),j) - nnetOuts(1:m ~= singlePredictions(i,j),j));
                end
            end
            
            % Obtain vote of most confident model for every sample
            [~,mostConfident] = max(confidence,[],1);
            predictions = singlePredictions(sub2ind(size(confidence),mostConfident,1:length(mostConfident)));
            
            predictions = epochVote(self,predictions)';
        end
        
        
        function predictions = epochVote(self,predictions)
            % Set all samples in epoch to same stage using mode filter
            L = length(predictions);
            for  i = 1:(L/self.epochLength)
                predictions(((i-1)*self.epochLength+1):(i*self.epochLength)) = mode(predictions(((i-1)*self.epochLength+1):(i*self.epochLength))); % In-place smoothing
            end
        end
    end
end