function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
disp('aqui size inside predict')
size(X)
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% =========================================================================


%Theta1 has size 25 x 401
%Theta2 has size 10 x 26
%X has size 5000x400, every row is a number example that has to be predicted
% we need to add the bias unit (add a 1 to every example in X) (X0 to x )
% com que despres trasposo X, realment afegir una columna de 1s a X es com haver ficat una fila de 1s,
% osigui haver afegit X0 que es el bias (s'ha de fer lo mateix per la hidden layer, afegir una fila a0)
X_rows = size(X,1)
X = [ones(X_rows,1) , X]
%Now X should be 5000x401
z2 = sigmoid(Theta1*X')
%z2 is size 25x5000
%now we should add add the bias for the hidden layer (z2)
z2 = [ones(1,size(z2,2)); z2]
%z2 is now size 26x5000
hx = sigmoid(Theta2*z2)
%hx is 10x5000
for i = 1:size(X,1)
[probabilitat, valorProposat] = max(hx(:,i),[],1)
p(i) = valorProposat
end
end