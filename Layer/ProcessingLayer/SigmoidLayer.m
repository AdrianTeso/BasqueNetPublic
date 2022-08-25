classdef SigmoidLayer < ProcessingLayer
    %DIGMOID Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function objSigmoidLayer = SigmoidLayer(name,size)
            %DIGMOID Construct an instance of this class
            %   Detailed explanation goes here
            objSigmoidLayer@ProcessingLayer(name, size);
        end
    end
    methods(Access = protected)
        function [output, layerObject] = internalProcess(layerObject, input)
            %PROCESS Apply the RELU operation on inputs
            %   Implementation of the class Layer  method
            output = zeros(layerObject.Size, 1);
            for iterator=1:layerObject.Size
                output(iterator) = 1/(1 + exp(-input(iterator)));
                if layerObject.Size > 1
                    layerObject.ProcessMatrix(iterator, iterator) = output(iterator) * (1 - output(iterator));
                else
                    layerObject.ProcessMatrix = output * (1 - output);
                end
            end
        end
    end
end

