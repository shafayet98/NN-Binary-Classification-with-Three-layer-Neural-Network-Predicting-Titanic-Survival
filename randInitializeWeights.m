function W = randInitializeWeights(L_in, L_out)

    W = zeros(L_out, 1 + L_in);
    epsilon_theta = 0.12;
    W = (rand(L_out,1+L_in) * 2 * epsilon_theta) - epsilon_theta;
    
end
