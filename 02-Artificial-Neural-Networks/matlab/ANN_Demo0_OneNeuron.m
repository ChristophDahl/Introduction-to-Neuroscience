%% ANN Demo 0 - How one artificial neuron makes a decision
% INTRODUCTION TO NEUROSCIENCE | Neural networks | 24 Nov 2026
% Base MATLAB only. Run this file in the MATLAB Editor.
% The two inputs are synthetic activity, NOT recorded neuronal firing.

clear; clc;

% x(1): activity of a left-preferring sensory population
% x(2): activity of a right-preferring sensory population
x = [0.8; 0.2];
w = [-2.0; 2.0];              % learned connection strengths, here chosen by hand
b = 0;                       % decision bias

z = w.' * x + b;             % weighted input
p_right = 1 / (1 + exp(-z)); % sigmoid output, between 0 and 1

fprintf('Weighted input z = %.2f\n', z);
fprintf('P(right) = %.3f\n', p_right);
if p_right >= 0.5
    fprintf('Decision: RIGHT\n');
else
    fprintf('Decision: LEFT\n');
end

% CLASSROOM QUESTIONS
% 1. Change x from [0.8; 0.2] to [0.2; 0.8]. What happens?
% 2. Change both weights to zero. What does the neuron predict?
% 3. Change b to +2. How does bias affect the decision?
% 4. Why is this NOT yet a learning demonstration?
