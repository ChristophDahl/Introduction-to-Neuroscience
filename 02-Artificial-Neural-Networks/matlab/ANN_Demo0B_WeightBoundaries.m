%% ANN Demo 0B — How weights and bias change a decision boundary
% Introduction to Neuroscience | 24 November 2026
% BASE MATLAB. Same saved 500-trial sample bank used by Python and slide 10.
% These rules have manually selected weights; there is NO training here.

clear; clc; close all;
nSamples = 160;       % TRY: 30, 160, or 500
sensoryNoise = 0.18;  % TRY: 0.08, 0.18, or 0.35

% Load the exact observations used to draw slide 10. A separate file
% stores true labels and unit-noise values; input noise is adjustable.
here = fileparts(mfilename('fullpath'));
file = fullfile(here,'..','data','ANN_demo0B_trials.csv');
M = dlmread(file,',',1,0);
assert(nSamples >= 2 && nSamples <= size(M,1), 'Choose between 2 and 500 trials');
M = M(1:nSamples,:);
yTrue = M(:,1);                  % 0 = CLASS A, 1 = CLASS B
X = zeros(nSamples,2);
X(:,1) = 0.72*(1-yTrue) + 0.24*yTrue + sensoryNoise*M(:,2);
X(:,2) = 0.24*(1-yTrue) + 0.72*yTrue + sensoryNoise*M(:,3);
X = min(1,max(0,X));             % numerical features scaled to [0,1]

% Each row: w1, w2, bias. Identical trials for every row.
cases = [-2,  2,   0;       % baseline
         -1,  3,   0;       % rotate boundary
         -2,  2,   0.8;     % shift boundary
         -6,  6,   0;       % steeper sigmoid, same decisions
         -0.5,0.5, 0;       % shallower sigmoid, same decisions
          2, -2,   0];     % reverse category sign
names = {'Baseline','Relative weights','Positive bias', ...
         'Large weights','Small weights','Reverse signs'};

%% Figure 1: matches the three panels of slide 10
figure('Color','w','Name','Slide 10: changing the separating line');
for k = 1:3
    subplot(1,3,k);
    plotRule(X,yTrue,cases(k,:),names{k});
end

%% Figure 2: additional weight manipulations on the SAME observations
figure('Color','w','Name','More weight settings');
for k = 4:6
    subplot(1,3,k-3);
    plotRule(X,yTrue,cases(k,:),names{k});
end

%% Figure 3: outputs change even when the decision boundary does not
figure('Color','w','Name','Same boundary, different sigmoid outputs');
d = linspace(-1,1,401);
scales = [0.5,2,6];
hold on;
for k = 1:numel(scales)
    p = 1./(1+exp(-scales(k)*d));
    plot(d,p,'LineWidth',2);
end
plot([-1 1],[0.5 0.5],'k:');
xlabel('Feature difference: x_2 - x_1');
ylabel('Sigmoid output for B');
legend({'Small weights','Baseline','Large weights'},'Location','southeast');
axis([-1 1 0 1]); grid on; hold off;

%% Explore
% 1. How does changing relative weights rotate the dividing line?
% 2. How does the bias move it without rotating it?
% 3. Why do the small, baseline and large weights classify identically?
% 4. What changes when you increase sensoryNoise or nSamples?
% 5. Has this neuron learned? No: all weights were assigned manually.

function plotRule(X,yTrue,oneCase,name)
    w = oneCase(1:2).'; b = oneCase(3);
    p = 1./(1+exp(-max(-30,min(30,X*w+b))));
    pred = double(p>=0.5);
    accuracy = 100*mean(pred==yTrue);
    ix0 = yTrue==0; ix1=yTrue==1;
    hold on;
    scatter(X(ix0,1),X(ix0,2),20,[.0 .58 .62],'filled');
    scatter(X(ix1,1),X(ix1,2),20,[.93 .43 .23],'filled');
    mistakes = pred~=yTrue;
    if any(mistakes)
        plot(X(mistakes,1),X(mistakes,2),'kx','LineWidth',.8,'MarkerSize',6);
    end
    xs = linspace(0,1,200);
    if abs(w(2))>eps
        ys = -(w(1)*xs+b)/w(2);
        plot(xs,ys,'k--','LineWidth',1.8);
    else
        xc = -b/w(1);
        plot([xc xc],[0 1],'k--','LineWidth',1.8);
    end
    axis([0 1 0 1]); axis square; grid on;
    xlabel('Feature 1 (x_1)'); ylabel('Feature 2 (x_2)');
    title(sprintf('%s: %.1f%%',name,accuracy));
    hold off;
    fprintf('%-18s w=[%+4.1f,%+4.1f], b=%+3.1f, test=%.1f%%\n', ...
        name,w(1),w(2),b,accuracy);
end
