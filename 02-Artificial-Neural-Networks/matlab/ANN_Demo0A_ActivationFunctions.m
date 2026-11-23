%% ANN Demo 0A - Compare six activation functions
% Introduction to Neuroscience | 24 Nov 2026
% BASE MATLAB ONLY. These curves transform the weighted sum z.
% Q: Which outputs can be interpreted as probabilities? Which are unbounded?

clear; clc; close all;
z = linspace(-5, 5, 501);
alpha = 0.10;                       % negative slope of Leaky ReLU; TRY 0.01 / 0.3

step = double(z >= 0);
sigmoid = 1 ./ (1 + exp(-z));
tanhOutput = tanh(z);
relu = max(0, z);
leakyRelu = max(alpha*z, z);
linearOutput = z;

names = {'Step', 'Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'Linear'};
curves = {step, sigmoid, tanhOutput, relu, leakyRelu, linearOutput};
limits = {[-0.1 1.1], [-0.1 1.1], [-1.2 1.2], [-0.5 5.2], [-1.2 5.2], [-5.2 5.2]};

figure('Color', 'w', 'Name', 'ANN Demo 0A: activation functions');
for k = 1:6
    subplot(2,3,k);
    plot(z, curves{k}, 'Color', [0.07 0.53 0.62], 'LineWidth', 2);
    hold on;
    plot([0 0], limits{k}, ':', 'Color', [0.6 0.6 0.6]);
    plot([-5 5], [0 0], ':', 'Color', [0.6 0.6 0.6]);
    hold off;
    xlim([-5 5]); ylim(limits{k});
    xlabel('Weighted input z'); ylabel('Output f(z)');
    title(names{k}); grid on;
end

% An identical positive z produces different outputs across functions.
zExample = 2;
fprintf('For z = %.1f:\n', zExample);
fprintf(' Step = 1; sigmoid = %.3f; tanh = %.3f; ReLU = %.1f; Leaky ReLU = %.1f\n', ...
    1/(1+exp(-zExample)), tanh(zExample), max(0,zExample), max(alpha*zExample,zExample));

% TRY:
% 1. Change alpha and re-run. What happens when z < 0?
% 2. Which activation functions can output a negative value?
% 3. Why is ReLU common in hidden layers but inappropriate as a probability?
% 4. A sigmoid output between 0 and 1 is NOT automatically calibrated.
