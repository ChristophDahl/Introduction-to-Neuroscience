%% Demo 3: context-dependent decisions (MATLAB, no add-on toolboxes)
% Synthetic task inspired by cognitive neuroscience; not real cortical data.
% Inputs: motion evidence, colour evidence, cue (1 = motion, 0 = colour).
% Target: report the sign of the evidence selected by the cue.
% Compare a linear sigmoid unit and a 10-unit tanh hidden-layer ANN.
clear; close all; clc; rng(22, 'twister');

nTrain = 800;
nTest = 400;
nEpochs = 1800;
learningRate = 0.4;
nHidden = 10;
noiseSD = 0.08;  % TRY: 0, 0.08, 0.25

% Independent training and test trials.
[Xtrain, yTrain, ~] = makeTrials(nTrain, noiseSD);
[Xtest, yTest, conflict] = makeTrials(nTest, noiseSD);
sigmoid = @(z) 1 ./ (1 + exp(-max(min(z,30),-30)));

%% Model 1: a single linear sigmoid neuron
w = zeros(3,1);
b = 0;
for epoch = 1:nEpochs
    p = sigmoid(Xtrain*w + b);
    err = p - yTrain;
    w = w - learningRate*(Xtrain'*err)/nTrain;
    b = b - learningRate*mean(err);
end

%% Model 2: nonlinear hidden layer trained by backpropagation
W1 = 0.5*randn(3,nHidden); b1 = zeros(1,nHidden);
W2 = 0.5*randn(nHidden,1); b2 = 0;
trainLoss = zeros(ceil(nEpochs/20),1);
for epoch = 1:nEpochs
    H = tanh(bsxfun(@plus, Xtrain*W1, b1));
    p = sigmoid(H*W2 + b2);
    if mod(epoch-1,20)==0
        q = min(max(p,1e-9),1-1e-9);
        trainLoss((epoch-1)/20+1) = -mean(yTrain.*log(q)+(1-yTrain).*log(1-q));
    end
    delta2 = (p-yTrain)/nTrain;
    gradW2 = H'*delta2;
    gradb2 = sum(delta2);
    delta1 = (delta2*W2').*(1-H.^2);
    gradW1 = Xtrain'*delta1;
    gradb1 = sum(delta1,1);
    W2 = W2 - learningRate*gradW2;
    b2 = b2 - learningRate*gradb2;
    W1 = W1 - learningRate*gradW1;
    b1 = b1 - learningRate*gradb1;
end

%% Test on unseen trials and then shuffle the task cue
pLinear = sigmoid(Xtest*w+b);
pHidden = sigmoid(tanh(bsxfun(@plus,Xtest*W1,b1))*W2 + b2);
XnoCue = Xtest;
XnoCue(:,3) = Xtest(randperm(nTest),3); % Break cue/target correspondence
pNoCue = sigmoid(tanh(bsxfun(@plus,XnoCue*W1,b1))*W2+b2);

acc = @(p,y) 100*mean(double(p>=0.5)==y);
aAll = [acc(pLinear,yTest),acc(pHidden,yTest),acc(pNoCue,yTest)];
aConflict = [acc(pLinear(conflict),yTest(conflict)), ...
             acc(pHidden(conflict),yTest(conflict)), ...
             acc(pNoCue(conflict),yTest(conflict))];

fprintf('CONTEXT-DEPENDENT DECISIONS (SYNTHETIC DATA)\n');
fprintf('Linear unit:       %.1f%%\n',aAll(1));
fprintf('Hidden-layer ANN:  %.1f%%\n',aAll(2));
fprintf('ANN, shuffled cue:   %.1f%%\n',aAll(3));
fprintf('Conflicting trials: %d / %d\n',sum(conflict),nTest);
fprintf('Accuracy on conflicting trials: %.1f%% / %.1f%% / %.1f%%\n',aConflict);

figure('Color','w','Name','ANN Demo 3: Context-dependent decisions');
subplot(1,2,1);
plot(0:20:nEpochs-1,trainLoss,'LineWidth',1.8);
xlabel('Training epoch'); ylabel('Cross-entropy loss');
title('Nonlinear network learns the task'); grid on;
subplot(1,2,2);
bar([aAll(:),aConflict(:)]);
ylim([0 105]); ylabel('Held-out accuracy (%)');
set(gca,'XTickLabel',{'Linear neuron','Hidden ANN','ANN, shuffled cue'});
xtickangle(15); legend({'All trials','Conflicting evidence'},'Location','southoutside');
hold on; plot([0.5 3.5],[50 50],':','Color',[0.4 0.4 0.4]); hold off;
title('Why task context matters'); grid on;

%% QUESTIONS
% 1. Why do linear and nonlinear models behave differently on conflicting trials?
% 2. Does better accuracy require stronger sensory inputs here?
% 3. Why does shuffling the cue reduce accuracy?
% 4. This feedforward model does NOT reproduce PFC recurrent dynamics.

function [X,y,conflict] = makeTrials(n,noiseSD)
    motion = (2*double(rand(n,1)>=0.5)-1).*(0.35+0.65*rand(n,1));
    colour = (2*double(rand(n,1)>=0.5)-1).*(0.35+0.65*rand(n,1));
    context = double(rand(n,1)>=0.5); % 1 motion, 0 colour
    y = double((context.*motion + (1-context).*colour)>0);
    X = [motion+noiseSD*randn(n,1), colour+noiseSD*randn(n,1),context];
    conflict = (motion>0)~=(colour>0);
end
