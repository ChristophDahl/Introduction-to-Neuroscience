% STDP classroom demonstration, following Bi & Poo (1998).
% Phenomenological model, NOT a fit to their experimental data.
% dt = t_post - t_pre: dt>0 is pre-before-post (potentiation).
clear; close all; clc;

N_pairs = 60;
w0 = 0.40;
w_min = 0;
w_max = 1;
A_plus = 0.005;
A_minus = 0.003;
tau_plus = 15;     % milliseconds
tau_minus = 15;    % milliseconds

dt = -60:0.25:60;
dw = stdp_increment(dt, A_plus, A_minus, tau_plus, tau_minus);
w_final = min(w_max, max(w_min, w0 + N_pairs * dw));

figure('Color','w', 'Position',[100 100 1100 700]);
subplot(2,2,1);
plot(dt, 100*(w_final-w0)/w0, 'LineWidth', 2); hold on;
xline(0,':'); yline(0,'--'); grid on;
xlabel('t_{post} - t_{pre} (ms)'); ylabel('Change in weight (%)');
title('A. Timing determines potentiation or depression');

subplot(2,2,2); hold on;
delays = [10 -10 40 -40];
for i = 1:numel(delays)
    increment = stdp_increment(delays(i), A_plus, A_minus, tau_plus, tau_minus);
    wt = min(w_max, max(w_min, w0 + (0:N_pairs)*increment));
    plot(0:N_pairs,wt,'LineWidth',2);
end
xlabel('Pairing number'); ylabel('Synaptic weight');
title('B. Same number of spikes, different order');
legend('\Deltat=+10 ms','\Deltat=-10 ms','\Deltat=+40 ms','\Deltat=-40 ms', 'Location','best'); grid on;

subplot(2,2,3); hold on;
window_widths = [5 15 30];
for i = 1:numel(window_widths)
    tau = window_widths(i);
    increment = stdp_increment(dt, A_plus, A_minus, tau, tau);
    wf = min(w_max, max(w_min, w0 + N_pairs * increment));
    plot(dt,100*(wf-w0)/w0,'LineWidth',2);
end
xlabel('Spike timing (ms)'); ylabel('Change in weight (%)');
title('C. Time-window width changes sensitivity');
legend('5 ms','15 ms','30 ms', 'Location','best'); yline(0,'--'); grid on;

subplot(2,2,4); hold on;
repeats = [10 30 60 100];
for i = 1:numel(repeats)
    wf = min(w_max,max(w_min,w0 + repeats(i)*dw));
    plot(dt,100*(wf-w0)/w0, 'LineWidth',2);
end
xlabel('Spike timing (ms)'); ylabel('Change in weight (%)');
title('D. Repetition changes magnitude');
legend('N=10','N=30','N=60','N=100', 'Location','best'); yline(0,'--'); grid on;

sgtitle('STDP — minimal model (not fitted to Bi & Poo, 1998)');
for d = [-40 -10 10 40]
    inc = stdp_increment(d,A_plus,A_minus,tau_plus,tau_minus);
    wf = min(w_max,max(w_min,w0+N_pairs*inc));
    fprintf('dt = %+d ms: final weight = %.3f\n',d,wf);
end

function dw = stdp_increment(dt,A_plus,A_minus,tau_plus,tau_minus)
    dw = zeros(size(dt));
    pos = dt > 0; neg = dt < 0;
    dw(pos) = A_plus*exp(-dt(pos)/tau_plus);
    dw(neg) = -A_minus*exp(dt(neg)/tau_minus);
    % Exactly simultaneous spikes are assigned zero for simplicity.
end
