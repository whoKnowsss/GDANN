function [Recall,Precision,Accuracy,F1score,Specificity,Kappa]=perf_ana(C)
% Syntax: [Recall,Precision,Accuracy,F1score,Specificity,Kappa]=perf_ana(C)
% This function yields several classification performance metrics based on 
% confusion matrix.
% Input - 
%   C: confusion matrix (predicted_labels x true_labels)
%   TP - C(1,1); FN - C(1,2); FP - C(2,1); TN - C(2,2)
% Output -
%   Accuracy, Precision, Recall (sensitivity), F1-score,
%   Specificity, and kappa are obtained.

TP=C(1,1); 
FN=C(1,2);
FP=C(2,1);
TN=C(2,2);

Recall=TP/(TP+FN);
Precision=TP/(TP+FP);
Accuracy=(TP+TN)/(TP+FN+FP+TN);
F1score=(2*Precision*Recall)/(Precision+Recall);
Specificity=TN/(TN+FP);

Total=(TP+FN+FP+TN);
Class1_C=(TP+FP); % Classifier output for class 1
Class1_GT=(TP+FN); % Ground truth for class 1
Class2_C=(TN+FN); % Classifier output for class 2
Class2_GT=(FP+TN); % Ground truth for class 2
ExpAcc=(((Class1_C*Class1_GT)/Total)+((Class2_C*Class2_GT)/Total))/Total;
Kappa=(Accuracy-ExpAcc)/(1-ExpAcc);

% Last modified by Monalisa Pal on 07/12/2016.
end