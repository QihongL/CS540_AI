%%
clear all; 
X = [7,8,10,20,26,30,32,33]';
Xlabels = {'6','8','10','20','26','30','32','33'};

% compute the tree
treeSingle = linkage(X,'single');
treeComplete = linkage(X,'complete');
treeAverage = linkage(X,'average');


%% 
FS = 16;
figure
% subplot(3,1,1)
[Ds, ~, Ds_outperm] = dendrogram(treeSingle,'Labels',Xlabels);
title('single linkage', 'fontsize', FS)
set(gca,'fontsize',FS); set(Ds,'LineWidth',2);

% subplot(3,1,2)
% [Dc, ~, Dc_outperm] = dendrogram(treeComplete,'Labels',Xlabels);
% title('complete linkage', 'fontsize', FS)
% set(gca,'fontsize',FS); set(Dc,'LineWidth',2);
% 
% subplot(3,1,3)
% [Da, ~, Da_outperm] = dendrogram(treeAverage,'Labels',Xlabels);
% title('average linkage', 'fontsize', FS)
% set(gca,'fontsize',FS); set(Da,'LineWidth',2);
