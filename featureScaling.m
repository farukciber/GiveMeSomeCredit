function [X] = featureScaling(X)

%Inicijalizacija mean i sigma u 0 za svaki column
mu = zeros(1,size(X,2));
sigma = zeros(1,size(X,2));

%Za svaki column izracunati mean i sigma i onda update
for i = 1:size(X,2)
    mu(i) = mean(X(:,i));
    sigma(i) = std(X(:,i));
    
    %Za svaki element u svakom columnu update
    X(:,i) =(X(:,i)-mu(i))/sigma(i);
end
end