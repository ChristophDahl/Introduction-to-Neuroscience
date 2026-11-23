%% Demo 4: learn handwritten digits using a small convolutional neural network
% The data are bundled, so NO internet is needed during the lecture.
% Requires MATLAB Deep Learning Toolbox; CPU training is sufficient.
% Dataset: scikit-learn/UCI Optical Recognition of Handwritten Digits,
% 1797 images; 8x8 grayscale; labels 0-9. NOT MNIST.
% Run from anywhere: paths are resolved relative to this .m file.
clear; close all; clc; rng(27,'twister');
if exist('convolution2dLayer','file')~=2
    error('ANN_Demo4_DigitsCNN requires MATLAB Deep Learning Toolbox.');
end
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    error('Save this file and execute it as a .m script (not pasted sections).');
end
thisDir = fileparts(scriptPath);
dataPath = fullfile(thisDir,'..','data','ANN_digits8x8.mat');
if ~isfile(dataPath)
    error('Dataset is missing: %s. Preserve the ../data folder.',dataPath);
end
S = load(dataPath);
X = single(S.images) / 16;   % 8 x 8 x 1 x 1797, values [0,1]
Y = categorical(double(S.labels(:)));
n = numel(Y);
idx = randperm(n);
nTrain = round(0.80*n);
idxTrain = idx(1:nTrain);idxTest = idx(nTrain+1:end);
XTrain = X(:,:,:,idxTrain); YTrain = Y(idxTrain);
XTest  = X(:,:,:,idxTest);  YTest  = Y(idxTest);
fprintf('Handwritten digits: %d train / %d test (8x8 pixels, ten classes)\n',nTrain,n-nTrain);

% 1. Illustrate inputs before training
figure('Color','w','Name','Digits database: example images', 'Position',[120 200 1000 390]);
for k = 1:20
    subplot(2,10,k);
    imagesc(X(:,:,1,idx(k)),[0 1]); axis image off; colormap(gray);
    title(char(Y(idx(k))));
end
sgtitle('Handwritten digit database: examples of 8x8 grayscale images');

% 2. Small CNN, with 3x3 convolutional kernels shared across image positions
layers = [
    imageInputLayer([8 8 1],'Normalization','none','Name','image')
    convolution2dLayer(3,8,'Padding','same','Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool')
    convolution2dLayer(3,16,'Padding','same','Name','conv2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(10,'Name','classifier')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','class_output')
];
options = trainingOptions('adam', ...
    'MaxEpochs',12,'MiniBatchSize',64,'InitialLearnRate',0.003, ...
    'Shuffle','every-epoch','Verbose',false,'Plots','training-progress', ...
    'ExecutionEnvironment','cpu');
net = trainNetwork(XTrain,YTrain,layers,options);

% 3. Evaluate on withheld test images
YPred = classify(net,XTest);
accuracy = 100*mean(YPred == YTest);
fprintf('Held-out accuracy: %.1f%% on %d images\n',accuracy,numel(YTest));
figure('Color','w','Name','CNN: predictions on new digit images', 'Position',[120 200 1100 475]);
for k=1:20
    subplot(2,10,k);
    imagesc(XTest(:,:,1,k),[0 1]); axis image off; colormap(gray);
    isCorrect=(YPred(k)==YTest(k));
    if isCorrect, c=[0 .45 .35]; else, c=[.80 .24 .20]; end
    title(sprintf('%s / %s',char(YPred(k)),char(YTest(k))), ...
        'Color',c,'FontWeight','bold');
end
sgtitle(sprintf('Predicted / actual digit | test accuracy %.1f%%',accuracy));
figure('Color','w','Name','CNN: confusion matrix');
confusionchart(YTest,YPred);title('Held-out digit classification');

% 4. Optionally inspect first-layer filters
weights = net.Layers(2).Weights;
figure('Color','w','Name','Learned first-layer CNN filters', 'Position',[150 150 850 340]);
for k=1:size(weights,4)
    subplot(2,4,k);imagesc(weights(:,:,1,k));axis image off;
    title(sprintf('Filter %d',k));colorbar;
end
sgtitle('Eight learned 3x3 kernels (weights shared across positions)');

% Questions: Why test on withheld images? What is shared about convolution?
% Why can the CNN still misclassify handwriting even after training?
