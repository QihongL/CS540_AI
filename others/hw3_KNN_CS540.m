clear all; close all; clc 
data = [90 52 7 10; 130 69 9.5 20; 50 45 6 10; 63 51.5 6.5 10; 145 70 11 20; 160 69.5 10 20];
% get X and y 
X = data(:,1:3);
y = data(:,4);
idx = [1:size(X,1)]';
% new points 
xnew1 = [100, 50, 7];
xnew2 = [120, 90, 9];

% compute distances
diff1 = bsxfun(@minus, X, xnew1);
distance1 = sum(diff1.^2,2);

horzcat(distance1, y)

diff2 = bsxfun(@minus, X, xnew2);
distance2 = sum(diff2.^2,2);
horzcat(distance2, y)