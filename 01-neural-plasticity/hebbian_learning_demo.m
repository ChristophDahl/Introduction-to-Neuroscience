%% Hebbian learning: paired versus unpaired activity
% Introductory neuroscience demonstration
% Learning rule: Delta w = eta * x * y
% x, y are binary neural activities (0 = inactive, 1 = active).

clear; close all; clc;

N   = 40;             % Number of trials
eta = 0.02;           % Learning rate
w0  = 0.20;           % Initial synaptic strength

% Neuron A is active on every second trial.
t = 1:N;
x = double(mod(t, 2) == 0);

% Two conditions with equal overall activity in neuron B:
y_paired   = x;        % A and B are active together
y_unpaired = 1 - x;    % A and B alternate; no overlap

% Record the synaptic weight after each trial.
w_paired   = zeros(1, N + 1);
w_unpaired = zeros(1, N + 1);
w_paired(1)   = w0;
w_unpaired(1) = w0;

for k = 1:N
    w_paired(k + 1)   = w_paired(k)   + eta * x(k) * y_paired(k);
    w_unpaired(k + 1) = w_unpaired(k) + eta * x(k) * y_unpaired(k);
end

%% Plot activity and learning
figure('Color','w','Position',[150 150 920 620]);

subplot(2,1,1);
imagesc([x; y_paired; y_unpaired]);
colormap(flipud(gray));   % Dark = active, light = inactive
set(gca,'YTick',1:3, 'YTickLabel', ...
    {'Neuron A', 'Neuron B (paired)', 'Neuron B (unpaired)'});
xlabel('Trial');
title('Neuronal activity (dark = active)');

subplot(2,1,2);
plot(0:N, w_paired, '-', 'LineWidth',2); hold on;
plot(0:N, w_unpaired, '--', 'LineWidth',2);
xlabel('Trial'); ylabel('Synaptic weight w');
title('Hebbian learning: \Delta w = \eta x y');
legend('Paired activity','Unpaired activity','Location','northwest');
grid on;

fprintf('Final weight, paired:   %.2f\n', w_paired(end));
fprintf('Final weight, unpaired: %.2f\n', w_unpaired(end));

% This model is intentionally simplified: it does not include
% spike timing, depression, weight saturation, or homeostasis.
