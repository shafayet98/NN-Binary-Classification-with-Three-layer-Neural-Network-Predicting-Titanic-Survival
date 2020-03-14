% load data
data_train = csvread("train_data.csv");
data_test = csvread("test_data.csv");

% extracting data from csv and creating training and test variable
X_train = data_train(2:length(data_train),4:17);
y_train = data_train(2:length(data_train),3);

X_test = data_test(2:length(data_test), 4:17);
y_test = data_test(2:length(data_test), 3);

% displaying the size of X, y and init_theta
% disp(size(X_train));
% disp(size(y_train));

input_layer = 14;
hidden_layer = 7;
output_layer = 1;

% randomly initialize weights
initial_Theta1 = randInitializeWeights(input_layer,hidden_layer);
initial_Theta2 = randInitializeWeights(hidden_layer,output_layer);

% unrolling the parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% calling the cost function to get the cost
% lamda = 1;
% [cost_J grad] = nnCostCalc(all_theta,input_layer,hidden_layer,output_layer,X_train,y_train,lamda);

%  using fmincg
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 1;

costFunction = @(p) nnCostFunction(p,input_layer,hidden_layer,output_layer, X_train, y_train, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% disp(nn_params);

Theta1 = reshape(nn_params(1:hidden_layer * (input_layer + 1)),hidden_layer, (input_layer + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer * (input_layer + 1))):end),output_layer, (hidden_layer + 1));


pred = predict(Theta1, Theta2, X_test);
for i = 1: length(pred),
    if pred(i) >= 0.5
        pred(i) = 1;
    elseif pred(i) < 0.5
        pred(i) = 0;
    endif;
endfor;

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
