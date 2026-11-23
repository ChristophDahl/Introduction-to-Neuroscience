%% ANN Demo 0E — Hebbian activity rule versus supervised gradient descent
% Base MATLAB, no toolboxes. Run after lecture slide 14.
%
% Both systems receive EXACTLY the same varying positive input x(k) and
% start at the same weight w(1). Each has one input and one linear output:
%   y(k) = w(k)*x(k).
%
% Hebbian:  w(new) = w + etaH * x * y
% Gradient: w(new) = w + etaG * (target-y) * x
%
% The external target is used ONLY by the supervised gradient rule.
% Hebbian learning is deliberately UNREGULATED here: biological/plausible
% models often include competition, decay, inhibition, or normalisation.
clear; close all; clc;

nSteps = 140;
k = 1:nSteps;
x = 0.65 + 0.30*sin((k-1)/11) + 0.10*cos((k-1)/4);
targetGain = 1.40;            % Teaching target: target(k) = 1.4 * x(k)
target = targetGain * x;

w0 = 0.40;                    % Same initial weight
etaH = 0.040;                 % Hebbian learning rate
etaG = 0.140;                 % Gradient-descent learning rate

wHebb = zeros(1,nSteps+1); wGrad = zeros(1,nSteps+1);
wHebb(1) = w0; wGrad(1) = w0;
yHebb = zeros(1,nSteps); yGrad = zeros(1,nSteps);
lossHebb = zeros(1,nSteps); lossGrad = zeros(1,nSteps);

for i = 1:nSteps
    % Same input, same output formula; different weights due to learning.
    yHebb(i) = wHebb(i) * x(i);
    yGrad(i) = wGrad(i) * x(i);

    % Measure both against an external reference, for comparison only.
    % This reference NEVER influences the Hebbian weight update.
    lossHebb(i) = 0.5 * (target(i)-yHebb(i))^2;
    lossGrad(i) = 0.5 * (target(i)-yGrad(i))^2;

    wHebb(i+1) = wHebb(i) + etaH * x(i) * yHebb(i);
    % dL/dw = (y-target)*x for L=0.5*(target-w*x)^2.
    gradient = (yGrad(i)-target(i))*x(i);
    wGrad(i+1) = wGrad(i) - etaG * gradient;
end

figure('Color','w','Name','Hebbian versus supervised learning',...
       'Position',[100 100 1100 650]);
subplot(2,2,1)
plot(k,x,'k','LineWidth',1.5); hold on
plot(k,target,'Color',[.1 .55 .6],'LineWidth',1.4)
legend('Input x(k)','Teaching target t(k)','Location','best')
ylabel('Activity / target'); title('Same inputs; target provided only to gradient rule')
xlim([1 nSteps]); grid on

subplot(2,2,2)
plot(0:nSteps,wHebb,'Color',[.85 .35 .12],'LineWidth',1.9); hold on
plot(0:nSteps,wGrad,'Color',[.08 .48 .72],'LineWidth',1.9)
plot([0 nSteps],[targetGain targetGain],'k:','LineWidth',1.2)
legend('Hebbian (unregulated)','Gradient descent','Target gain','Location','northwest')
ylabel('Weight w'); title('How the weights evolve'); xlim([0 nSteps]); grid on

subplot(2,2,3)
plot(k,target,'k:','LineWidth',1.8); hold on
plot(k,yHebb,'Color',[.85 .35 .12],'LineWidth',1.4)
plot(k,yGrad,'Color',[.08 .48 .72],'LineWidth',1.6)
legend('Target','Hebbian output','Gradient output','Location','northwest')
xlabel('Training step'); ylabel('Output y'); title('The same input produces different outputs')
xlim([1 nSteps]); grid on

subplot(2,2,4)
plot(k,lossHebb,'Color',[.85 .35 .12],'LineWidth',1.6); hold on
plot(k,lossGrad,'Color',[.08 .48 .72],'LineWidth',1.6)
legend('Hebbian evaluated against target','Gradient descent','Location','northwest')
xlabel('Training step'); ylabel('Loss L = 1/2(t-y)^2')
title('Only gradient descent uses the target to learn')
xlim([1 nSteps]); grid on

fprintf('Initial w = %.2f; target gain = %.2f\n',w0,targetGain);
fprintf('Final Hebbian w = %.3f; gradient-descent w = %.3f\n',wHebb(end),wGrad(end));
fprintf('Mean loss (last 20 steps): Hebb %.4f; gradient %.6f\n',...
    mean(lossHebb(end-19:end)),mean(lossGrad(end-19:end)));

% Questions for students:
% 1) Does the Hebbian rule know the desired output? No.
% 2) What happens if etaH is set to 0? What if it is raised to 0.08?
% 3) Set etaG = 0.01. How quickly does the supervised learner converge?
% 4) Why does an unregulated positive Hebbian rule keep increasing w?
% 5) Can Hebbian learning be stabilised by other mechanisms? (Yes.)
