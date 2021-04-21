function test_labels = recognize(classifier, test_data)
% Syntax: test_labels = recognize(classifier, test_data)
% This function uses a trained classifier to predict the labels of 
% instances of the dataset under test called test_data. 
% Input - 
%   classifier: Structure variable representing a trained classifier
%   test_data:  Testing dataset (matrix order #instances x #features)
% Output - 
%   test_labels: Predicted labels (column vector where #entries 
%        corresponding to the #instances of test_data is stored)

%test_labels = svmclassify(classifier, test_data);
% Last modified by Monalisa Pal on 07/12/2016.

     
        test_labels = predict(classifier, test_data);
 
end