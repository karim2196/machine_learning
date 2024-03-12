function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


anomalies = pval < epsilon
%if I sum elements 1 on both vectors, i got a new vector where the value "2"
%means that i summed "1" in anomalies and "1" in y val
% then, i want 1s at the positions where there is a "2" and sum them to count the elements "1"
% that match in both vectors
tp = sum ( (anomalies + yval) == 2)
fp = 0
fn = 0
%-------------ITERATIVE SOLUTION-------------
%anomaliesAndYval = anomalies + yval == 1
%for i = 1:size(anomaliesAndYval,2)
%    if ( ( anomaliesAndYval(i) == anomalies(i) ) && ( anomaliesAndYval(i) != yval(i) ) )
%        fp = fp + 1
%    endif
%    if (anomaliesAndYval(i) == yval(i) && anomaliesAndYval(i) != anomalies(i))
%        fn = fn + 1
%    endif

%end
%-------------BETTER SOLUTION (VECTORIZED) USING LOGICAL AND-------------
fp = sum ( (anomalies == 1) & (yval == 0) )
fn = sum ( (anomalies == 0) & (yval == 1) )
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1 = (2 * precision * recall) / (precision + recall)

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
