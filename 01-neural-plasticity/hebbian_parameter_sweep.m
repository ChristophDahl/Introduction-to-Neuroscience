%% HEBBIAN LEARNING — SYSTEMATIC PARAMETER SWEEP
% Introduction to Neuroscience | Teaching demonstration
%
% Simple rate/binary Hebbian rule:
%    w(t+1) = w(t) + eta * x(t) * y(t)
%
% Five systematically varied factors:
%   eta  : learning rate
%   p    : firing fraction of EACH neuron (identical for both neurons)
%   c    : fraction of presynaptic spikes coinciding with postsynaptic spikes
%          c = P(y=1 | x=1), for these constructed binary activity patterns
%   N    : number of trials
%   w0   : initial synaptic weight
%
% WHY THIS DESIGN?
% When c changes, both neurons fire equally often (p*N times each).
% This isolates co-activation from overall activity level.
% Restriction p <= 0.5 permits c=0 (fully nonoverlapping firing).
%
% IMPORTANT LIMITATIONS:
%  * This rule has no LTD, weight saturation, homeostatic regulation, or STDP.
%  * With nonnegative activities, weights cannot decrease.
%  * c ~ p is the expected coincidence for independent activity, not c = 0.
%  * Spike order does NOT matter here: only simultaneous co-activation does.
%  * w0 shifts final weight but does not affect the weight CHANGE.
%
% MATLAB R2016b+ (local function at end of script); no toolboxes required.

clear; close all; clc;
rng(7);                          % Reproducible activity ordering

%% 1. Factorial design: 3 x 2 x 5 x 3 x 2 = 90 conditions
eta_values = [0.005, 0.020, 0.050];
p_values   = [0.20, 0.40];             % Activity fraction for EACH neuron
c_values   = [0, 0.25, 0.50, 0.75, 1.00];
N_values   = [40, 80, 160];
w0_values  = [0.20, 0.50];

outdir = fullfile(pwd, 'hebbian_outputs');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

n_total = numel(eta_values)*numel(p_values)*numel(c_values)* ...
          numel(N_values)*numel(w0_values);
rows = zeros(n_total, 9);
row_id = 0;

for a = 1:numel(eta_values)
    for b = 1:numel(p_values)
        for cc = 1:numel(c_values)
            for d = 1:numel(N_values)
                for e = 1:numel(w0_values)
                    eta = eta_values(a);
                    p   = p_values(b);
                    c   = c_values(cc);
                    N   = N_values(d);
                    w0  = w0_values(e);

                    % Exactly p*N spikes for x AND y; exactly c*p*N overlaps.
                    [x,y] = make_activity(N,p,c);

                    w = zeros(N+1,1);
                    w(1) = w0;
                    for t = 1:N
                        w(t+1) = w(t) + eta*x(t)*y(t);
                    end

                    gain = w(end)-w0;
                    c_observed = sum(x.*y)/sum(x);
                    predicted_gain = eta*N*p*c;
                    assert(abs(gain-predicted_gain)<1e-10, ...
                           'Unexpected mismatch: check integer trial counts.');

                    row_id = row_id+1;
                    rows(row_id,:) = [eta,p,c,N,w0,gain,w(end),c_observed, ...
                                      predicted_gain];
                end
            end
        end
    end
end

results = array2table(rows, 'VariableNames', ...
    {'eta','p','c','N','w0','weight_change','final_weight', ...
     'observed_c','predicted_change'});
writetable(results,fullfile(outdir,'hebbian_factorial_results.csv'));
fprintf('Completed %d conditions.\n',height(results));
fprintf('All results saved to: %s\n',outdir);

%% 2. Figure 1 — Experimental logic, time courses, and interactions
f1 = figure('Color','w','Position',[100 100 1190 780]);

% A. Weight trajectories at different co-activation levels
subplot(2,2,1); hold on;
eta=0.02; p=0.40; N=80; w0=0.20;
c_demo = [0, 0.50, 1];
colors = lines(numel(c_demo));
for k=1:numel(c_demo)
    [x,y] = make_activity(N,p,c_demo(k));
    w = [w0, w0+eta*cumsum(x.*y)];
    stairs(0:N,w,'LineWidth',2,'Color',colors(k,:));
end
xlabel('Trial'); ylabel('Synaptic weight');
title('A. Vary co-activation; hold firing rate constant');
legend({'c = 0','c = 0.5','c = 1'},'Location','northwest');
grid on; box off;

% B. Example input spikes with partial coactivation
subplot(2,2,2); hold on;
[x,y] = make_activity(40,0.40,0.50);
stem(find(x==1),ones(1,sum(x)),'filled','LineWidth',1.0, ...
     'MarkerSize',3,'Color',[0.22 0.46 0.78]);
stem(find(y==1),2*ones(1,sum(y)),'filled','LineWidth',1.0, ...
     'MarkerSize',3,'Color',[0.85 0.40 0.18]);
ylim([0.4,2.5]); xlim([0,41]);
set(gca,'YTick',[1,2],'YTickLabel',{'Pre (x)','Post (y)'});
xlabel('Trial'); title('B. Same activity rate; some spikes coincide');
grid on; box off;

% C. Heatmap: learning rate x coactivation
subplot(2,2,3);
eta_grid = eta_values(:);
c_grid = c_values;
Z = (eta_grid * c_grid) * (80*0.40);  % Delta w = eta * N * p * c
imagesc(c_grid,eta_grid,Z); axis xy;
set(gca,'XTick',c_values,'YTick',eta_values);
xlabel('Co-activation c'); ylabel('Learning rate \eta');
title('C. Weight change: learning rate x co-activation');
colorbar;

% D. Effect of practice (N) and firing rate p
subplot(2,2,4); hold on;
for k=1:numel(p_values)
    vals = 0.20 + 0.02*N_values*p_values(k)*0.75;
    plot(N_values,vals,'-o','LineWidth',2,'MarkerSize',7);
end
xlabel('Number of trials (N)'); ylabel('Final synaptic weight');
title('D. Practice x activity rate (c = 0.75)');
legend({'p = 0.2','p = 0.4'},'Location','northwest');
grid on; box off;

set(f1,'PaperPositionMode','auto');
saveas(f1,fullfile(outdir,'figure1_hebbian_parameter_effects.png'));

%% 3. Figure 2 — Initial weight is NOT a driver of weight change
f2 = figure('Color','w','Position',[100 100 1120 420]);

subplot(1,2,1); hold on;
for k=1:numel(c_values)
    gains = 0.02*80*0.40*c_values(k);
    plot(w0_values,w0_values+gains,'-o','LineWidth',1.5);
end
xlabel('Initial synaptic weight (w_0)');
ylabel('Final synaptic weight');
title('E. Initial weight shifts the outcome');
legend({'c=0','c=.25','c=.5','c=.75','c=1'},'Location','northwest');
grid on; box off;

subplot(1,2,2); hold on;
for k=1:numel(c_values)
    gains = 0.02*80*0.40*c_values(k);
    plot(w0_values,gains*ones(size(w0_values)),'-o','LineWidth',1.5);
end
xlabel('Initial synaptic weight (w_0)');
ylabel('Weight change (\Delta w)');
title('F. Initial weight does NOT change learning');
grid on; box off;

set(f2,'PaperPositionMode','auto');
saveas(f2,fullfile(outdir,'figure2_initial_weight.png'));

%% 4. Compact analytic check for teaching
% In this particular controlled binary design:
%   sum_t x(t)y(t) = N * p * c
% so the ENTIRE experiment reduces to:
%   w_final = w0 + eta * N * p * c
fprintf('\nAnalytic prediction: w_final = w0 + eta*N*p*c\n');
fprintf('Example: w0=.2, eta=.02, N=80, p=.4, c=.5 -> w_final=%.2f\n', ...
        0.2+0.02*80*0.4*0.5);
fprintf('Note: independence means c is APPROXIMATELY p on average, not c=0.\n');

%% Local function — controlled binary activity patterns
function [x,y] = make_activity(N,p,c)
    nx = round(N*p);
    no = round(nx*c);
    assert(abs(nx-N*p)<1e-10 && abs(no-nx*c)<1e-10, ...
        'Choose N, p, c so spike counts are integers.');
    assert(p<=0.5, 'For arbitrary c=0 at equal rates, p must not exceed 0.5.');

    x = zeros(1,N);
    y = zeros(1,N);
    active = randperm(N,nx);
    x(active)=1;

    % Put no spikes of y at locations already active in x.
    if no>0
        idx_shared = active(randperm(nx,no));
        y(idx_shared)=1;
    end
    remaining_y = nx-no;
    if remaining_y>0
        inactive = find(x==0);
        idx_other = inactive(randperm(numel(inactive),remaining_y));
        y(idx_other)=1;
    end
end
