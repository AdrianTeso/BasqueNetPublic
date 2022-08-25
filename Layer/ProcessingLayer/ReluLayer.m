classdef ReluLayer < ProcessingLayer
    %RELULAYER Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function reluLayerObject = ReluLayer(name, size)
            %RELULAYER Construct an instance of this class
            %   Detailed explanation goes here
            reluLayerObject@ProcessingLayer(name, size);
            reluLayerObject.Size = size;
            
        end
    end
    methods(Access = protected)
        function [output, layerObject] = internalProcess(layerObject, input)
            %PROCESS Apply the RELU operation on inputs
            %   Implementation of the class Layer  method
            if(length(input) ~= layerObject.Size)
                output = zeros(layerObject.Size, 1);
            else
                output = zeros(layerObject.Size,1);
                for iterator=1:layerObject.Size
                    if input(iterator) < 0
                        output(iterator) = 0;
                        layerObject.ProcessMatrix(iterator, iterator) = 0;
                    else
                        output(iterator) = input(iterator);
                        layerObject.ProcessMatrix(iterator, iterator) = 1;
                    end
                end
            end
        end
        
    end
end

