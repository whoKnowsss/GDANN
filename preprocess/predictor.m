function classifier = predictor(train_data, train_labels)
% Syntax: classifier = predictor(train_data, train_labels,type)
% This function trains the classifier using training data. Here, Support 
% Vector Machine is used with linear kernel and boxconstraint as the 
% soft margin. 
% Input - 
%   train_data:  Training dataset (matrix order #instances x #features)
%   train_labels: Training labels (column vector where #entries 
%        corresponding to the #instances of train_data is stored)
% Output - 
%   classifier: Structure variable representing the trained classifier
%option=statset('MaxIter',100000);
%classifier = svmtrain(train_data, train_labels, 'kernel_function',...
%    'linear', 'boxconstraint',1,'options',option);
%% Last modified by Monalisa Pal on 07/12/2016.

%options.MaxIter = 500000;
%classifier = svmtrain(train_data, train_labels, 'kernel_function',...
%    'linear', 'boxconstraint',1,'options',options);

        classifier = fitcsvm(train_data, train_labels, 'KernelFunction',...
  'linear', 'BoxConstraint',1);
end