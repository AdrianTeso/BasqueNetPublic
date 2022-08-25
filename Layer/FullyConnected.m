classdef FullyConnected < Layer
    %FULLYCONNNECTED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        InputSize
        OutputSize
        WeightsMatrix
        BiasMatrix
        Seed
        LastInput
        LastOutput
        LastMomentum
        LastMomentumBias
        WeightSum
        DropOutRate
        GuardadoMediaPesos
        GuardadoMediaBias
        LastModification
        LastModificationBias
        LastWeightMatrix
        HaveResidual
    end
    
    methods
        function fullyConnectedObject = FullyConnected(name,inputSize, outputSize, seed, dropOut, residual)
            %FULLYCONNNECTED Construct an instance of this class
            %   Detailed explanation goes here
            if(nargin < 5)
                dropOut = 0;
            end
            if(nargin < 6)
                residual = false;
            end
            fullyConnectedObject@Layer(name);
            fullyConnectedObject.InputSize = inputSize;
            fullyConnectedObject.OutputSize = outputSize;
            fullyConnectedObject.Seed = seed;
            rng(fullyConnectedObject.Seed);
            fullyConnectedObject.WeightsMatrix = random('Uniform',-1,1,outputSize,inputSize)./outputSize;
            fullyConnectedObject.BiasMatrix = random('Uniform',-1,1,outputSize,1)./outputSize;
            fullyConnectedObject.LastMomentum = zeros(outputSize,inputSize);
            fullyConnectedObject.LastMomentumBias = zeros(outputSize,1);
            fullyConnectedObject.LastModification = zeros(outputSize,inputSize);
            fullyConnectedObject.LastModificationBias = zeros(outputSize,1);
            fullyConnectedObject.DropOutRate = dropOut;
            fullyConnectedObject.GuardadoMediaPesos = zeros(outputSize,inputSize);
            fullyConnectedObject.GuardadoMediaBias = zeros(outputSize,1);
            fullyConnectedObject.LastWeightMatrix = fullyConnectedObject.WeightsMatrix;
            fullyConnectedObject.HaveResidual = residual;
        end
        function fullyConnectedObject = eraseLastModification(fullyConnectedObject)
            fullyConnectedObject.LastModification = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.LastModificationBias = zeros(fullyConnectedObject.OutputSize,1);
        end
        function fullyConnectedObject = resizeInputs(fullyConnectedObject, newSize)
            fullyConnectedObject.InputSize = newSize;
            fullyConnectedObject.WeightsMatrix =[];
            fullyConnectedObject.LastMomentum = [];
            fullyConnectedObject.LastModification = [];
            fullyConnectedObject.GuardadoMediaPesos =[];
            fullyConnectedObject.LastWeightMatrix =[];
            fullyConnectedObject.WeightsMatrix = random('Uniform',-1,1,fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize)./fullyConnectedObject.OutputSize;
            fullyConnectedObject.LastMomentum = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.LastModification = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.GuardadoMediaPesos = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.LastWeightMatrix = fullyConnectedObject.WeightsMatrix;
        end
        function fullyConnectedObject = resizeOutputs(fullyConnectedObject, newSize)
            fullyConnectedObject.OutputSize = newSize;
            fullyConnectedObject.WeightsMatrix = [];
            fullyConnectedObject.BiasMatrix = [];
            fullyConnectedObject.LastMomentum = [];
            fullyConnectedObject.LastMomentumBias = [];
            fullyConnectedObject.LastModification = [];
            fullyConnectedObject.LastModificationBias = [];
            fullyConnectedObject.GuardadoMediaPesos = [];
            fullyConnectedObject.GuardadoMediaBias = [];
            fullyConnectedObject.LastWeightMatrix = [];

            fullyConnectedObject.WeightsMatrix = random('Uniform',-1,1,fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize)./fullyConnectedObject.OutputSize;
            fullyConnectedObject.WeightsMatrix = fullyConnectedObject.WeightsMatrix .* eye(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.BiasMatrix = random('Uniform',-1,1,fullyConnectedObject.OutputSize,1)./fullyConnectedObject.OutputSize;
            fullyConnectedObject.LastMomentum = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.LastMomentumBias = zeros(fullyConnectedObject.OutputSize,1);
            fullyConnectedObject.LastModification = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.LastModificationBias = zeros(fullyConnectedObject.OutputSize,1);
            fullyConnectedObject.GuardadoMediaPesos = zeros(fullyConnectedObject.OutputSize,fullyConnectedObject.InputSize);
            fullyConnectedObject.GuardadoMediaBias = zeros(fullyConnectedObject.OutputSize,1);
            fullyConnectedObject.LastWeightMatrix = fullyConnectedObject.WeightsMatrix;
        end
    end
    methods(Access = protected)
        function [output, layerObject] = internalProcess(layerObject, input)
            %PROCESS Summary of this method goes here
            %   Detailed explanation goes here
            layerObject.LastWeightMatrix = layerObject.WeightsMatrix;
            if isequal(size(input),[layerObject.InputSize, 1])
                output = layerObject.WeightsMatrix * double(input) + ...
                    layerObject.BiasMatrix;
            else
                output = zeros(layerObject.OutputSize, 1);
            end
            layerObject.LastInput =input;
            layerObject.LastOutput = output;
        end
        function partDev = internalBackward(layerObject, previouLayerBackward)
            partDev = previouLayerBackward * layerObject.LastWeightMatrix;
        end
        function kaparenDeribatua = internalJakobianoaLortu(layerObject)
            kaparenDeribatua = layerObject.LastWeightMatrix;
        end
        
    end
end

