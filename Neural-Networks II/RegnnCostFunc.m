function [J grad] = RegnnCostFunc(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%here goes cost function

%FIRST WE NEED TO TRANSFORM EVERY Y OUTPUT (0,1,2,3...) TO BINARY
yBinary = zeros(size(y,1),num_labels)
for i=1:size(y,1)
    numberToDecode = y(i) % from 0 to 9
    numberInBinary = zeros(1,num_labels)
    numberInBinary(numberToDecode) = 1
    disp('----------------------number to decode--------------')
    disp(numberToDecode)
    disp('---------------------- binary decoded----------------')
    disp(numberInBinary)
end
%HERE ENDS THE BINARY TRANFORMATION
%X is size 5000x400 (no bias unit, we should do that now)
X = [ones(size(X,1),1),X]
z2 = X*Theta1'
%z2 is size 5000x25, (no bias unit, we should do that now to z2)
z2 = [ones(size(z2,1),1),z2]
%z2 is size 5000x26
a2 = sigmoid(z2)
z = Theta2*a2'
%z is size 10x5000
hx = sigmoid(z)
%haig de guardar 10 cost functions, 1 per cada k unit i fer el sum

for j = 1:num_labels
    kElemOfAllY =yBinary(:,j) 
    costFuncVectorized = (-kElemOfAllY' * log(hx'(:,j))) - ((1-kElemOfAllY')*log(1-hx'(:,j)))
end

for l = 1:num_labels
    costFuncOverK = sum (costFuncVectorized(l,:) )
end

costFuncOverM = sum(costFuncRow)

firstPartOfSum = 1/m * costFuncOverM

% =========================================================================
%Calculating regularization for theta 1
thetaResultAux = zeros(size(Theta1,1),1)
for row = 1:size(Theta1,1)
    thetaResultAux(row) = sum(Theta1(row,2:end))
end
allTheta1 = sum(thetaResultAux)

%Calculating regularization for theta 1
thetaResultAux = zeros(size(Theta2,1),1)
for row = 1:size(Theta2,1)
    thetaResultAux(row) = sum(Theta2(row,2:end))
end
allTheta2 = sum(thetaResultAux)

%calculating final regularization term
regularizationTerms = (lambda/(2*m)) * (allTheta1+allTheta2)

%calculating J regularized
J  = ( (1/m) * firstPartOfSum ) + regularizationTerms
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
