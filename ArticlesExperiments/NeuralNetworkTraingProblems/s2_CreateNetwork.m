
addpath('BasqueNet');
addpath('LearningFunctions');
addpath('Layer');
addpath('Layer\ProcessingLayer');
addpath('AuxiliaryFuctions')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create the network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bn = BasqueNet("Neural Network Traing Problems");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create the learning parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
bn = setLearningParameters(bn,0.01,MSEIkasketa,100, 0.01, 10, 1, "SGD", true,false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add the necessary layer to the network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bn = addLayer(bn,FullyConnected("X5",2,100,1,0.2, false));
bn = addLayer(bn,ReluLayer("X4", 100));
bn = addLayer(bn,FullyConnected("X3",100,20,1,0.2, false));
bn = addLayer(bn,ReluLayer("X2", 20));
bn = addLayer(bn,FullyConnected("X1",20,1,1,0.2));


