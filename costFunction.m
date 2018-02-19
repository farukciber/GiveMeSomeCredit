function [cost, gradientFinal] = costFunction(theta, X, y, lambda)

%Length od m
m = length(y);

%Inicijalizovanje novih theta
gradient = zeros(size(theta));

%Hipoteza
hypothesis = sigmoid(X*theta);

%Cost funkcija sa regularizacijom
cost = (-1/m) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis)) + lambda/(2*m)*sum(theta.^2);

%Gradient descent bez regularizacije
for i = 1:m
	gradient = gradient + ( hypothesis(i) - y(i) ) * X(i, :)';
end

%Regularizacijski dio gradienta
gradientRegularization = lambda/m*[0;theta(2:end)];

%Spojena 2 dijela gornja
gradientFinal = (1/m) * gradient + gradientRegularization;

end