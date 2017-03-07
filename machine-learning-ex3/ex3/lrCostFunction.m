function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_t = theta;
theta_t(1) = 0;

J = (-y' * log(sigmoid(X*theta)) - (1-y)'*log(1 - sigmoid(X*theta))) / m;
J = J + (lambda * sum(theta_t.^2) / (2*m));

grad = X' * (sigmoid(X*theta)-y ) / m;
grad = grad + (lambda*theta_t / m);

% =============================================================

grad = grad(:);

end
