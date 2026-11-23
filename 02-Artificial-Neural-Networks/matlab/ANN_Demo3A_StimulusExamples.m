%% Demo 3A: illustrative colour/motion dot stimuli
% Corresponds to the *sign convention* in Demo 3, not the actual input data.
% In Demo 3 the model receives only numerical motion/colour evidence and cue.
% Positive motion => RIGHT; negative => LEFT.
% Positive colour => BLUE; negative => RED.
% Congruent: both evidence signs agree. Conflict: signs differ.
% No toolboxes required.
clear; close all; clc; rng(19,'twister');
N = 90;
caseNames = {'RIGHT + BLUE','LEFT + RED','RIGHT + RED','LEFT + BLUE'};
evidence = [ 0.7  0.7; -0.7 -0.7; 0.7 -0.7; -0.7 0.7];
figure('Color','w','Name','Illustrative motion and colour stimuli', ...
    'Position',[100 200 1250 345]);
for k = 1:4
    subplot(1,4,k); hold on;
    motion = evidence(k,1); colour = evidence(k,2);
    a = 2*pi*rand(N,1); r = sqrt(rand(N,1));
    x = r.*cos(a); y = r.*sin(a);
    isBlue = false(N,1);isBlue(randperm(N,round(N*(1+colour)/2))) = true;
    isRight = false(N,1);isRight(randperm(N,round(N*(1+motion)/2))) = true;
    scatter(x(isBlue),y(isBlue),24,[0.17 0.50 0.74],'filled');
    scatter(x(~isBlue),y(~isBlue),24,[0.84 0.35 0.23],'filled');
    selected = randperm(N,12);
    for j = selected
        dx = (2*isRight(j)-1)*0.15;
        quiver(x(j),y(j),dx,0,0,'Color',[.16 .20 .28], ...
            'LineWidth',1.1,'MaxHeadSize',2);
    end
    ang = linspace(0,2*pi,240);plot(cos(ang),sin(ang),'k','LineWidth',1);
    axis equal off; xlim([-1.1 1.1]);ylim([-1.1 1.1]);
    title(caseNames{k},'FontWeight','bold');
    if sign(motion)==sign(colour)
        text(0,-1.25,'CONGRUENT','HorizontalAlignment','center', ...
            'Color',[0 .45 .45],'FontWeight','bold');
    else
        text(0,-1.25,'CONFLICTING','HorizontalAlignment','center', ...
            'Color',[.70 .30 .15],'FontWeight','bold');
    end
end
sgtitle('Dot illustrations only: Demo 3 trains on numerical evidence, not images');
