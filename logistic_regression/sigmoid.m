function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z)); % g is a vector with the size of z (if matrix : n_rows , n_columns are returned)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
dimensions = size(g)
for iterI = 1:dimensions(1)
    for iterJ = 1:dimensions(2)
        disp('------------')
        disp(iterI)
        disp(iterJ)
        disp('------------')
        g(iterI,iterJ) = (1/(1+e^-z(iterI,iterJ)))
       
    end
end
% =============================================================
disp('resultat sigmoid : ')
disp(g)
end

