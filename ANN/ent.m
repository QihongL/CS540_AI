clear variables; close all; 
total = 100; 

i = 0; 
for freqA = 1 : total-1;
    
    x = [freqA, total - freqA];
    pmf = x ./ sum(x);
    
    i = i + 1; 
    ent(i) = sum(-pmf .* log(pmf)/log(2));
end

plot(ent)
title('Entropy over balanceness (2 classes)')
xlabel('Frequency of 1 class')
ylabel('Entropy')
