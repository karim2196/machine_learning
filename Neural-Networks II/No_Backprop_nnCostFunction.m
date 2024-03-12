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
    yBinary(i,:) = numberInBinary
end
%HERE ENDS THE BINARY TRANFORMATION
%X is size 5000x400 (no bias unit, we should do that now)
X = [ones(size(X,1),1),X]
z2 = X*Theta1'
%z2 is size 5000x25, (no bias unit, we should do that now to z2)
a2 = sigmoid(z2)
a2 = [ones(size(a2,1),1),a2]
z = a2*Theta2'
%z is size 5000x10
hx = sigmoid(z)
%haig de guardar 10 cost functions, 1 per cada k unit i fer el sum
costFuncVectorized = zeros(size(yBinary,1),num_labels)
kElemOfAllY = zeros(size(yBinary,1),1)
for j = 1:num_labels
    kElemOfAllY =yBinary(:,j)
    %kelem' is 1x5000, hx is 5000x1
    costFuncVectorized(:,j) = (-kElemOfAllY .* log(hx(:,j))) - ((1-kElemOfAllY).*log(1-hx(:,j)))
end

costFuncOverK = zeros(size(yBinary,1),1)
for l = 1:size(yBinary,1)
    costFuncOverK(l) = sum (costFuncVectorized(l,:) )
end

costFuncOverM = sum(costFuncOverK)

%J = 1/m * costFuncOverM

%Calculating regularization for theta 1
%theta square element wise 
Theta1Squared = Theta1.^2
Theta2Squared = Theta2.^2
thetaResultAux = zeros(size(Theta1Squared,1),1)
for row = 1:size(Theta1Squared,1)
    thetaResultAux(row) = sum(Theta1Squared(row,2:end))
end
allTheta1 = sum(thetaResultAux)

%Calculating regularization for theta 2
thetaResultAux = zeros(size(Theta2Squared,1),1)
for row = 1:size(Theta2Squared,1)
    thetaResultAux(row) = sum(Theta2Squared(row,2:end))
end
allTheta2 = sum(thetaResultAux)

%calculating final regularization term
regularizationTerms = (lambda/(2*m)) * ( (allTheta1) + (allTheta2) )

%calculating J regularized
J  = ( (1/m) * costFuncOverM ) + regularizationTerms

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end