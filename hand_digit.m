%% Hand_Digit
%%
clc;
clear;
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;
number_of_tests = 200;
lambda=1;


%% train
%%
clc;
%loading data_set
load('ex4data1.mat');
% buliding our trainig matrix
train = [X(1:300,:)' X(501:800,:)' X(1001:1300,:)' X(1501:1800,:)' X(2001:2300,:)' X(2501:2800,:)' X(3001:3300,:)' X(3501:3800,:)' X(4001:4300,:)' X(4501:4800,:)']'; 
y_train = [y(1:300)' y(501:800)' y(1001:1300)' y(1501:1800)' y(2001:2300)' y(2501:2800)' y(3001:3300)' y(3501:3800)' y(4001:4300)' y(4501:4800)']';
% visualizing data
visualize(train,50);
title('our training set');
% initializing weigths
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% building cost function and applying back_prop


costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, train, y_train, lambda);
 
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
   
% reshaping weigths

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
%% test 
%%
clc;
t =  [2877,2566,5000;189,4988,3555]; % by changing t you change the test number
test = X(t,:);  

test = reshape(test,size(t,1)*size(t,2),400);

a1=[ones(size(test,1),1) test];
a2=sigmoid(a1*Theta1');
a3=[ones(size(a2,1),1) a2];
final=(a3*Theta2');
testdigits = sigmoid(final);
[row , col] = find(testdigits > 0.5);
l=find(col==10);
col(l) = 0;
visualize(test,size(t,2));
str = string(col);
title("digit is " +str,'color','b');
             
   
