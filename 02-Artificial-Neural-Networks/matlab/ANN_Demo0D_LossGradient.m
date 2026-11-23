%% ANN Demo 0D: A complete numerical example of gradient descent
% Base MATLAB. A single linear unit: input x=1, bias b=0, target t=2.
% Output y = w*x; loss L = 0.5*(target-y)^2.
clear; clc; close all;

target = 2;
x = 1;
eta = 0.50;        % Try 0.10, 0.50, 1.00, 1.50, 2.20
nSteps = 7;
w = zeros(nSteps+1,1);
loss = zeros(nSteps+1,1);
for k = 1:nSteps+1
    y = w(k)*x;
    loss(k) = 0.5*(target-y)^2;
    fprintf('Step %d: w=%.4f, y=%.4f, loss=%.4f\n',k-1,w(k),y,loss(k));
    if k <= nSteps
        grad = (y-target)*x;   % derivative of this particular loss wrt w
        w(k+1) = w(k)-eta*grad;
    end
end

ww = linspace(-0.5,3.5,401);
figure('Color','w','Name','Gradient descent on a simple loss curve');
plot(ww,0.5*(target-ww*x).^2,'LineWidth',2); hold on;
plot(w,loss,'o-','LineWidth',1.8,'MarkerSize',7);
xlabel('Weight w'); ylabel('Loss L'); title('Moving towards lower prediction error');
grid on; legend('Loss curve','Successive updates','Location','north');
