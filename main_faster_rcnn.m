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

%% Display one img
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Create Faster R-CNN network
inputSize = [224 224 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData ,numAnchors);

featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

numClasses = width(cupDataset)-1;

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%% Data augmentation
% augmentedTrainingData = transform(trainingData,@augmentData);

%% Preprocess training dat
% trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
% validationData = transform(validationData,@(data)preprocessData(data,inputSize));

trainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);
    %% Train Faster R-CNN
    options = trainingOptions('sgdm',...
        'MaxEpochs',2,...
        'MiniBatchSize',2,...
        'InitialLearnRate',1e-3,...
        'CheckpointPath',tempdir,...
        'ValidationData',validationData);


    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);

%% Testing the data

figure();
randIdx = randi([1,400],1,6);
for i=1:6
    subplot(2,3,i);
    I = imread(cupDataset.cupImagename{randIdx(i)});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);

    if length(scores) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
        imshow(I)
    else
       bboxes = [0 0 100 100];
       scores = 'Not found';
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       imshow(I)
    end
end

figure();
randIdx = randi([1,400],1,6);
for i=1:6
    subplot(2,3,i);
    I = imread(cupDataset.cupImagename{randIdx(i)});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);

    if length(scores) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
        imshow(I)
    else
       bboxes = [0 0 100 100];
       scores = 'Not found';
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       imshow(I)
    end
end

figure();
randIdx = randi([1,400],1,6);
for i=1:6
    subplot(2,3,i);
    I = imread(cupDataset.cupImagename{randIdx(i)});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);

    if length(scores) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
        imshow(I)
    else
       bboxes = [0 0 100 100];
       scores = 'Not found';
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       imshow(I)
    end
end

figure();
randIdx = randi([1,400],1,6);
for i=1:6
    subplot(2,3,i);
    I = imread(cupDataset.cupImagename{randIdx(i)});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);

    if length(scores) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
        imshow(I)
    else
       bboxes = [0 0 100 100];
       scores = 'Not found';
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       imshow(I)
    end
end

figure();
randIdx = randi([1,400],1,6);
for i=1:6
    subplot(2,3,i);
    I = imread(cupDataset.cupImagename{randIdx(i)});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);

    if length(scores) ~= 0
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
        imshow(I)
    else
       maxIdx = maxIdx + 1; 
    end
end

%% Precision graph
    testData = transform(testData,@(data)preprocessData(data,inputSize));
    detectionResults = detect(detector,testData,'MinibatchSize',10);   
    [ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);

    figure
    plot(recall,precision)
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf('Average Precision = %.2f', ap))

