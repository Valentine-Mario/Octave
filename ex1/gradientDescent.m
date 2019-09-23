function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    %extract x1 form the data set
    x=X(:, 2);
    %get your hypothesis
    h=theta(1) + (theta(2) *x);
    %calculate the value for theta 1, here x1=0
    theta_zero=theta(1)-alpha*((1/m)* sum(h-y));
    %calculate value for theta 2
    thetaone=theta(2)- alpha* 1/m * sum((h-y).*x);
    %create a new vector for theta one and thetha two
    theta=[theta_zero; thetaone];
    
    % You need to return the following variables correctly 
    
    
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration
    %append the value of the cost func to the zero vector above    
   J_history(iter) = computeCost(X, y, theta);

end
  disp(min(J_history));
end
