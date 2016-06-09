clear all; clc; close all; 
X = [4,5;
    6,6;
    5,10;
    7,5;
    7,10;
    3,4;
    2,7;
    4,2;];

% cc = [0,0;2,5];
cc = [7,10; 5,5];
%% plot 
hold on 
plot(cc(:,1),cc(:,2), '*', 'linewidth', 5)
plot(X(:,1),X(:,2), 'x', 'linewidth', 2)
ylim([0,11])
xlim([0,10])
hold off