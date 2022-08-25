classdef SoftMaxLayer < ProcessingLayer
    %SOFTMAXLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function softMaxLayerObject = SoftMaxLayer(name, size)
            %SOFTMAXLAYER Construct an instance of this class
            %   Detailed explanation goes here
            softMaxLayerObject@ProcessingLayer(name,size);
        end
    end
    methods(Access = protected)
        function [output, layerObject] = internalProcess(layerObject, input)
            %PROCESS Apply the RELU operation on inputs
            %   Implementation of the class Layer  method
            output = zeros(layerObject.Size, 1);
            divisor = 0;
            for iterator = 1: length(input)
               divisor = divisor + exp(input(iterator));
               output(iterator) = exp(input(iterator));
            end
            if divisor ~= 0
                output = output / divisor;
            end
            layerObject.ProcessMatrix = diag(output) - (output .* output);
        end
        
    end
end

