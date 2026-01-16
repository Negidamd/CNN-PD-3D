clear all;close all;clc
%% Please run this file after the 1st training pipeline and 
%% re-RUN this file for two to three times to improve the accuracy 
%% Data Loading 
mriRootDataFolder = 'Dataset_Updated';
mriDataFolder = fullfile(mriRootDataFolder, 'Processed images - 2nd session');
allFiles = dir(fullfile(mriDataFolder));
fileDir  = fullfile({allFiles.folder},{allFiles.name});
srcDir_Healthy   = fileDir(3:27);
srcDir_Parkinson  = fileDir(28:48);
%% DataStore creation based for two Label  (Healthy and Parkinson)
imdsHealthy = imageDatastore(srcDir_Healthy, 'FileExtensions', '.nii');
imdsParkinson = imageDatastore(srcDir_Parkinson, 'FileExtensions', '.nii');

imds = imageDatastore({srcDir_Healthy{:},srcDir_Parkinson{:}}, 'FileExtensions', '.nii',...
    'ReadFcn',@niftiread,'ReadSize', 10);
imds.Labels=categorical([repmat({'Healthy'},size(imdsHealthy.Files));repmat({'Parkinson'},size(imdsParkinson.Files))]);

%% Loading 3D Resnet18() Model and changing layers as per our requirement
load 3DPretrainedModel
% if there would be error then uncomment this line 
mriNet = layerGraph(mriNet);

%mriNet = resnet18TL3Dfunction();
% numClasses = numel(categories(imds.Labels));
% newLearnableLayer = fullyConnectedLayer(numClasses, ...
%     'Name','new_fc', ...
%     'WeightLearnRateFactor',10, ...
%     'BiasLearnRateFactor',10);
% mriNet = replaceLayer(mriNet,'fc1000',newLearnableLayer);
% newClassLayer = classificationLayer('Name','new_classoutput');
% mriNet = replaceLayer(mriNet,'ClassificationLayer_predictions',newClassLayer);
inputSize = mriNet.Layers(1).InputSize;

%% Augmentation and Data Splitting in Training, Testing and Validation
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,.7,.1);
augimdsTrain = transform(imdsTrain,@(data,info)classification3DAugmentationPipeline(data,info,inputSize,'train'),'IncludeInfo',true);
augimdsValidation = transform(imdsValidation,@(data,info)classification3DAugmentationPipeline(data,info,inputSize,'validation'),'IncludeInfo',true);
augimdsTest = transform(imdsTest,@(data,info)classification3DAugmentationPipeline(data,info,inputSize,'test'),'IncludeInfo',true);
%% Train Model with Options
trainOpts.initLearnRate   = 0.001; % 10x reduction in initial learning rate
trainOpts.valFrequency    = 4; 
trainOpts.miniBatchSize   = floor(numel(augimdsTrain.UnderlyingDatastore.Files)/trainOpts.valFrequency); % rounding needed in case offline data augmentation is disabled 
trainOpts.maxEpochs       = 15;

options = trainingOptions('sgdm', ...
    'MiniBatchSize',trainOpts.miniBatchSize, ...
    'MaxEpochs',trainOpts.maxEpochs, ...
    'InitialLearnRate',trainOpts.initLearnRate, ...
    'Shuffle','every-epoch', ... % this handles the case where the mini-batch size doesn't evenly divide the number of training images
    'ValidationData',augimdsValidation, ... % source of validation data to evaluate learning during training
    'ValidationFrequency',trainOpts.valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress'); % display a plot of progress during training

mriNet = trainNetwork(augimdsTrain,mriNet,options);
%% Saving the updated pretrained model in Root Directory for further use 
save ('3DPretrainedModel.mat', 'mriNet')