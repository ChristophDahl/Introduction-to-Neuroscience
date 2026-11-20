%% ANN Demo 2 - Why a nonlinear hidden layer matters (XOR)
% INTRODUCTION TO NEUROSCIENCE | Neural networks | 24 Nov 2026
% Requirements: base MATLAB; NO Statistics or Deep Learning Toolbox.
% The four binary examples are TRAINING cases; this is not a generalization test.
% The network is trained with explicit gradient-descent / backpropagation steps.

clear; clc; close all;
X = [0 0; 0 1; 1 0; 1 1];
t = [0; 1; 1; 0];
n = size(X, 1);
eta = 1.0;
nEpochs = 7000;

%% First: train one linear sigmoid neuron
wLinear = zeros(2, 1);
bLinear = 0;
for epoch = 1:nEpochs
    p = 1 ./ (1 + exp(-(X*wLinear + bLinear)));
    delta = (p-t)/n;
    wLinear = wLinear - eta*(X.'*delta);
    bLinear = bLinear - eta*sum(delta);
end
pLinear = 1 ./ (1 + exp(-(X*wLinear+bLinear)));

%% Next: train a small network with ONE nonlinear hidden layer
% Architecture: two inputs --> four tanh hidden units --> one sigmoid output
% Use explicit initial values so every classroom run is reproducible.
W1 = [0.6, -0.9, 0.5, 1.2; 0.8, 0.7, -1.1, -0.5];
b1 = zeros(1, 4);
W2 = [0.3; -0.5; 0.8; -0.2];
b2 = 0;
loss = zeros(nEpochs, 1);

for epoch = 1:nEpochs
    H = tanh(X*W1 + repmat(b1, n, 1));
    p = 1 ./ (1 + exp(-max(-30, min(30, H*W2+b2))));
    pSafe = max(1e-10, min(1-1e-10, p));
    loss(epoch) = -mean(t.*log(pSafe) + (1-t).*log(1-pSafe));

    % Gradient passed backward through output and hidden layer.
    deltaOut = (p-t)/n;
    gradW2 = H.'*deltaOut;
    gradb2 = sum(deltaOut);
    deltaHidden = (deltaOut*W2.').*(1-H.^2);
    gradW1 = X.'*deltaHidden;
    gradb1 = sum(deltaHidden, 1);

    W1 = W1 - eta*gradW1;
    b1 = b1 - eta*gradb1;
    W2 = W2 - eta*gradW2;
    b2 = b2 - eta*gradb2;
end

H = tanh(X*W1 + repmat(b1, n, 1));
pNetwork = 1 ./ (1 + exp(-(H*W2 + b2)));

fprintf('DEMO 2 - XOR\n');
fprintf('x1 x2 | target | one neuron | hidden layer\n');
for k = 1:n
    fprintf(' %d  %d |   %d    |   %.3f    |    %.3f\n', ...
        X(k,1), X(k,2), t(k), pLinear(k), pNetwork(k));
end
fprintf('Linear training accuracy: %.0f %%\n',100*mean((pLinear>=0.5)==t));
fprintf('Network training accuracy: %.0f %%\n',100*mean((pNetwork>=0.5)==t));
fprintf('NOTE: these are the 4 training cases, not an independent test set.\n');

%% Plot training error and the learned nonlinear boundary
figure('Color','w','Name','Demo 2: XOR with a hidden layer');
subplot(1,2,1);
plot(loss,'Color',[0.08 0.38 0.65],'LineWidth',2);
xlabel('Training epoch');ylabel('Cross-entropy loss');
title('Hidden-layer network learns XOR');grid on;

subplot(1,2,2); hold on;
axis([-.3 1.3 -.3 1.3]);
[xGrid,yGrid] = meshgrid(linspace(-.25,1.25,100));
Q = [xGrid(:), yGrid(:)];
HQ = tanh(Q*W1 + repmat(b1,size(Q,1),1));
pQ = 1 ./ (1 + exp(-(HQ*W2+b2)));
contour(xGrid,yGrid,reshape(pQ,size(xGrid)),[.5 .5], ...
        'LineWidth',2,'LineColor',[.12 .35 .65]);
scatter(X(t==0,1),X(t==0,2),130,[.09 .55 .7],'o','filled');
scatter(X(t==1,1),X(t==1,2),130,[.95 .52 .22],'s','filled');
xlabel('Input x_1');ylabel('Input x_2');
title('Nonlinear decision boundary');axis square;grid on; hold off;

% QUESTIONS
% 1. Can one straight line separate the two XOR response classes?
% 2. What changes when we add hidden units and a nonlinear activation?
% 3. Is 100% on four TRAINING cases proof of good generalization? (No.)
% 4. Try decreasing the number of hidden units from 4 to 1. Modify W1,
%    b1, W2 consistently. What happens to the representational capacity?
% IMPORTANT: ANN hidden layers need not correspond to anatomical cortex layers.
