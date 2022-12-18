clc; clear; close all;

% trainNew = false;
% % model = 'model_RCNN_1';
% showPrecision = false;

%% Load dataset
load('cupDataset.mat')
I = imread(cupDataset.cupImagename{1});
boxes = cupDataset.cup{1};
I = insertObjectAnnotation(I,'rectangle',boxes,'');
figure
imshow(I)

%% Create train test data
rng(0)
shuffledIndices = randperm(height(cupDataset));
idx = 200; % Pocet vzoriek na trenovanie

trainingIdx = 1:idx;
trainingDataTbl = cupDataset(shuffledIndices(trainingIdx),:);

valNum = 100; % Pocet vzoriek na validaciu
validationIdx = idx+1 : idx + 1 + valNum;
validationDataTbl = cupDataset(shuffledIndices(validationIdx),:);

% Zvysok ide na testovanie
testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = cupDataset(shuffledIndices(testIdx),:);

%% Create datastores
imdsTrain = imageDatastore(trainingDataTbl{:,'cupImagename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'cup'));

imdsValidation = imageDatastore(validationDataTbl{:,'cupImagename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'cup'));

imdsTest = imageDatastore(testDataTbl{:,'cupImagename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'cup'));

%% Combine image data and box labels
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%% Create YOLOv2 Detector
inputSize = [224 224 3];
numClasses = 1;

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));

numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
preprocessedTrainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
data = read(trainingData);
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir,...
        'ValidationData',validationData)
%% Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
    ownImages=imageDatastore("cup_own_test\")
%% Evaluation of detector
    preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
%% 


figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))