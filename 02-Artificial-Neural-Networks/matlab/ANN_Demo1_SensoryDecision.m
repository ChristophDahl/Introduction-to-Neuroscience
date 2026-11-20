%% ANN Demo 1 - Learning a decision from noisy sensory inputs
% INTRODUCTION TO NEUROSCIENCE | Neural networks | 24 Nov 2026
% Requirements: base MATLAB; NO Statistics or Deep Learning Toolbox.
% This is a synthetic two-input sensory readout (not recorded MT neurons).
% Run the entire script. Change 'sensoryNoise' to 0.2, 0.6, 1.2.

clear; clc; close all;
rng(12, 'twister');          % reproducible random trials
sensoryNoise = 0.6;         % TRY: 0.2, 0.6, 1.2
nTrain = 240;
nTest = 120;
learningRate = 0.4;
nEpochs = 400;

%% Make independent training and test trials
% Label 0 = leftward motion; label 1 = rightward motion.
% Left/right-preferring populations are simple synthetic Gaussian variables.
yTrain = double(rand(nTrain, 1) > 0.5);
XTrain = sensoryNoise * randn(nTrain, 2);
XTrain(:, 1) = XTrain(:, 1) + 0.9 * (1 - yTrain);
XTrain(:, 2) = XTrain(:, 2) + 0.9 * yTrain;

yTest = double(rand(nTest, 1) > 0.5);
XTest = sensoryNoise * randn(nTest, 2);
XTest(:, 1) = XTest(:, 1) + 0.9 * (1 - yTest);
XTest(:, 2) = XTest(:, 2) + 0.9 * yTest;

%% Train a single artificial decision neuron
% p(right) = sigmoid(X*w + b)
% Error-correcting gradient descent minimizes binary cross-entropy.
% The gradient is supplied; students do NOT need to derive it.
w = zeros(2, 1);
b = 0;
loss = zeros(nEpochs, 1);

for epoch = 1:nEpochs
    p = 1 ./ (1 + exp(-max(-30, min(30, XTrain*w + b))));
    pSafe = max(1e-10, min(1-1e-10, p));
    loss(epoch) = -mean(yTrain.*log(pSafe) + (1-yTrain).*log(1-pSafe));

    errorSignal = p - yTrain;
    gradW = (XTrain.' * errorSignal) / nTrain;
    gradB = mean(errorSignal);
    w = w - learningRate * gradW;
    b = b - learningRate * gradB;
end

%% Evaluate on trials that were NOT used during training
pTest = 1 ./ (1 + exp(-max(-30, min(30, XTest*w + b))));
predTest = double(pTest >= 0.5);
accuracy = mean(predTest == yTest);
fprintf('DEMO 1 - NOISY SENSORY DECISIONS\n');
fprintf('Noise level: %.2f\n', sensoryNoise);
fprintf('Weights: left = %+.3f, right = %+.3f; bias = %+.3f\n', w(1), w(2), b);
fprintf('Test accuracy: %.1f %% (chance expectation = 50 %%)\n', 100*accuracy);

%% Figure: prediction error and decision boundary
figure('Color', 'w', 'Name', 'Demo 1: learning a sensory decision');
subplot(1, 2, 1);
plot(1:nEpochs, loss, 'LineWidth', 2, 'Color', [0.08 0.38 0.65]);
xlabel('Training epoch'); ylabel('Cross-entropy loss');
title('Training reduces prediction error');
grid on;

subplot(1, 2, 2); hold on;
leftIdx = yTest == 0; rightIdx = yTest == 1;
scatter(XTest(leftIdx, 1), XTest(leftIdx, 2), 30, [0.09 0.55 0.7], 'filled');
scatter(XTest(rightIdx, 1), XTest(rightIdx, 2), 30, [0.95 0.52 0.22], 'filled');
limits = [min(XTest(:, 1))-0.2, max(XTest(:, 1))+0.2];
xs = linspace(limits(1), limits(2), 150);
if abs(w(2)) > 1e-9
    ys = -(w(1)*xs + b) / w(2);
    plot(xs, ys, 'k--', 'LineWidth', 2);
end
xlabel('Left-preferring input (synthetic)');
ylabel('Right-preferring input (synthetic)');
title(sprintf('Unseen trials: %.1f%% correct', 100*accuracy));
legend({'Left trials', 'Right trials', 'Decision boundary'}, 'Location', 'best');
axis square; grid on; hold off;

% If desired: saveas(gcf, 'ANN_Demo1_Result.png');
% QUESTIONS
% 1. What happens to held-out accuracy when sensoryNoise increases?
% 2. Does changing the noise change the sensory signal, the weights, or both?
% 3. What do the signs of the two learned weights mean?
% 4. Why do we evaluate on new trials rather than only on training data?
% NEUROSCIENCE CAUTION: this supervised rule is NOT a model of Hebbian LTP.
