%% ANN Demo 0C: Time-varying activation in two artificial units
% Base MATLAB. No Deep Learning Toolbox required.
% THIS FIRST DEMO IS NOT TRAINING: a weight change is prescribed at t=61.
% The learning rules that determine weights from data come later.
clear; close all; clc;

nSteps = 120;
t = 1:nSteps;                % Discrete simulation steps, not biological time
x = 0.75 + 0.25*sin(2*pi*(t-1)/24); % The SAME input for both units
b = -1.20;                  % Same fixed bias for both units
wA = 0.80*ones(size(t));
wA(t >= 61) = 2.40;         % Prescribed change in unit A's weight
wB = 0.80*ones(size(t));    % Unit B: unchanged-weight control

zA = wA.*x + b;
zB = wB.*x + b;

fNames = {'Sigmoid (0 to 1)', 'tanh (-1 to 1)', 'ReLU', 'Leaky ReLU'};
fA = {1./(1+exp(-zA)), tanh(zA), max(zA,0), max(zA,0.1*zA)};
fB = {1./(1+exp(-zB)), tanh(zB), max(zB,0), max(zB,0.1*zB)};

figure('Color','w','Name','Same input, changing weights');
subplot(3,1,1);
plot(t,x,'k','LineWidth',1.7); ylabel('Input x(t)'); grid on;
title('A shared time-varying input'); xlim([t(1),t(end)]);
subplot(3,1,2);
plot(t,wA,'b-','LineWidth',2); hold on;
plot(t,wB,'r--','LineWidth',1.8); ylabel('Weight w(t)');
legend('Unit A: weight changes','Unit B: fixed weight','Location','best');
grid on; xlim([t(1),t(end)]); ylim([0.4,2.8]);
subplot(3,1,3);
plot(t,zA,'b-','LineWidth',1.8); hold on;
plot(t,zB,'r--','LineWidth',1.6); plot([t(1),t(end)],[0,0],'k:');
ylabel('Weighted sum z(t)'); xlabel('Simulation step');
grid on; xlim([t(1),t(end)]);

figure('Color','w','Name','Different activation functions');
for k = 1:4
    subplot(2,2,k);
    plot(t,fA{k},'b-','LineWidth',1.8); hold on;
    plot(t,fB{k},'r--','LineWidth',1.6);
    limits=ylim; plot([61,61],limits,'k:','LineWidth',1.0);
    title(fNames{k}); xlabel('Simulation step'); ylabel('Output y(t)');
    xlim([t(1),t(end)]); grid on;
    if k == 1, legend('Changed weight','Fixed weight','Location','best'); end
end

sigA=fA{1}; reluA=fA{3};
fprintf('At t=60: wA=%.1f, x=%.3f, zA=%.3f, sigmoid=%.3f, ReLU=%.3f\n', ...
    wA(60),x(60),zA(60),sigA(60),reluA(60));
fprintf('At t=61: wA=%.1f, x=%.3f, zA=%.3f, sigmoid=%.3f, ReLU=%.3f\n', ...
    wA(61),x(61),zA(61),sigA(61),reluA(61));

% Student exercises:
% 1) Change the step from t=61 to t=80.
% 2) Change the new weight from 2.40 to 1.40.
% 3) Change b from -1.20 to -0.50. Which units become active in ReLU?
% 4) Compare response ranges. Which functions can produce negative outputs?
