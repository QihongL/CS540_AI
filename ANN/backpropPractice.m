%% MLP simulation - learning logic gates
% this script shows that perceptron can learn AND and OR, but not XOR
clear variables; clc; close all;

% set learning parameters
gateType = 'xor';
alpha = .1;
maxEpoch = 50000;
minEpoch = 100; 
method = 'SGD';
nHidden = 2;

%% initialization
% choose the learning task
[INPUTS, TARGET] = genLogicGateAnswer(gateType);

% read param
m = size(INPUTS,1);     % num training examples
n = size(INPUTS,2);     % num features
% get input feature vectors (combine with bias)
X = horzcat(INPUTS, ones(m,1))';

% init weights
wts.ih = rand(nHidden,n+1) * 0.05;
wts.ho =  rand(1,nHidden+1) * 0.05;

%% training
epoch = 0; 
while true;
    epoch = epoch + 1;
    err = zeros(m,1);
    
    %% Full gradient decent
%     act.hidden = sigmoid(wts.ih * X);
%     act.out = sigmoid(wts.ho * [act.hidden; ones(1,m)]);
    
    %% stochastic gradient decent with randomly permuated order
    order = randperm(m);
    for i = order;
        % forward prop
        act.hidden = sigmoid(wts.ih * X(:,i));
        act.out = sigmoid(wts.ho * [act.hidden; 1]);
        % back prop
        delta.out = (act.out-TARGET(i))*act.out*(1-act.out);
        delta.hidden = delta.out * wts.ho(1:nHidden)' .* (act.hidden .* (1-act.hidden));
        % update weights
        wtsChange.ho = delta.out * [act.hidden;1]';
        wtsChange.ih = delta.hidden * X(:,i)';
        wts.ho = wts.ho - alpha * wtsChange.ho;
        wts.ih = wts.ih - alpha * wtsChange.ih;
        % record error 
        err(i) = act.out - TARGET(i);
        temp.change.ih(i) = mean(abs(wtsChange.ih(:)));
        temp.change.ho(i) = mean(abs(wtsChange.ho(:)));
    end
    
    % record error for the current iterations 
    errTotal(epoch) = sum(abs(err));
    change.h1(epoch) = mean(temp.change.ih);
    change.h2(epoch) = mean(temp.change.ho);
    
    % stopping criteria
    if all(err < .2)
        break;
    end
end


%% show final output
act.hidden = sigmoid(wts.ih * X);
act.out = sigmoid(wts.ho * [act.hidden; ones(1,m)]);
horzcat(TARGET, round(act.out)')
errTotal(end)

%% plot the learning curve
FS = 14;
LW = 2; 
subplot(1,2,1)
plot(errTotal, 'linewidth',LW)
title_text = sprintf('Learning curve with %s, alpha = %.2f, iter = %d',  ...
    method, alpha, epoch);
title(title_text, 'fontsize', FS)
xlabel('Epoch', 'fontsize', FS);ylabel('Error', 'fontsize', FS);

% plot the magnitude of weight update
subplot(1,2,2)
hold on 
plot(change.h1, 'linewidth',LW)
plot(change.h2, 'linewidth',LW)
hold off 
legend({'1st hidden layer','2nd hidden layer'},...
    'location', 'northeast', 'fontsize', FS)
title('Compare weight change magnitude', 'fontsize',FS)
xlabel('Epoch', 'fontsize', FS);
ylabel('Average absolute magnitude of weight change', 'fontsize', FS);
