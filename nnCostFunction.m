function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)


    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));

    m = size(X, 1);
    J = 0;
    Theta1_grad = zeros(size(Theta1)); % 7X15
    Theta2_grad = zeros(size(Theta2)); % 1X8

    % Forward propagation
    X = [ones(m,1) X];
    a1 = X; % 792X15
    z2 = Theta1 * a1' ; % 7X15 * 15X792  = 7X792
    a2 = sigmoid(z2);   % 7X792
    a2 = a2';           % 792X7
    a2 = [ones(length(a2),1) a2]; % 792X8
    a3 = Theta2 * a2' ; % 1X8 * 8X792 = 1X792;
    a3 = sigmoid(a3);   % 1X792;

    
    yNew = y'; % y=792X1 --> y' = 1X792

    J = (1/m) * sum( sum( (-yNew .* log(a3)) - ( (1-yNew) .* log(1-a3)   )  ) );

    % Regularized Cost Function Calculation

    t1 = Theta1(:,2:size(Theta1,2)); % 7X14
    t2 = Theta2(:,2:size(Theta2,2)); % 1X7

    reg = ((lambda) * ( (sum(sum(t1.^2))) + (sum(sum(t2.^2))) ))/ (2*m);
    J = J + reg;



    % Back Propagation
    for t = 1:m,

        % step - 01
        a1 = X(t,:); % 1X15
        a1 = a1' ; % 15X1
        z2 = Theta1 * a1 ; % 7X15 * 15X1 = 7X1
        a2 = sigmoid(z2); % 7X1
        a2 = [1 ; a2]; % 8X1

        z3 = Theta2 * a2 ; % 1X8 8X1 = 1X1
        a3 = sigmoid(z3); % 1X1

        % step - 02
        del_3 = a3 - yNew(:,t); % 1X1 - 1X1 = 1X1;

        % step - 03
        z2 = [1;z2]; % 8X1
        del_2 = ( Theta2' * del_3 ) .* sigmoidGradient(z2) ; % 8X1 * 1X1 = 8X1 
        del_2  = del_2(2:end); % 7X1

        % step - 04
        Theta1_grad = Theta1_grad + (del_2 * a1') ; % 7X15 + ( 7X1 * 1X15 ) = 7X15 + 7X15 = 7X15
        Theta2_grad = Theta2_grad + (del_3 * a2') ; % 1X8 + ( 1X1 * 1X8 ) = 1X8 + 1X8 =  1X8

    endfor;

    % step - 05
    Theta1_grad = Theta1_grad ./ m ; % 7X15
    Theta2_grad = Theta2_grad ./ m ; % 1X8

    % Regularized

    % Theta1_grad(:,1) = Theta1_grad(:,1) ./ m ; % we do not regularize the bias term of Theta 1; 
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end)); % non bias terms
    % Theta2_grad(:,1) = Theta2_grad(:,1) ./ m ; % we do not regularize the bias term of Theta 2;
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end)); % non bias terms


    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
