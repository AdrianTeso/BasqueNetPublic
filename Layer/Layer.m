classdef Layer
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Name
        
    end
    
    methods
        function layerObject = Layer(name)
            %LAYER Construct an instance of this class
            %   Detailed explanation goes here
            layerObject.Name = name;
        end
        function [output, layerObject] = process(layerObject, input)
            [output, layerObject] = internalProcess(layerObject, input);
        end
        function partDev = backward(layerObject, previouLayerBackward)
            partDev = internalBackward(layerObject, previouLayerBackward);
        end
        function kaparenDeribatua = jakobianosLortu(layerObject)
            kaparenDeribatua = internalJakobianoaLortu(layerObject);
        end
    end
    methods(Access = protected)
        function [output, layerObject] = internalProcess(layerObject, input)
            output = input;
            layerObject = layerObject;
        end
        function partDev = internalBackward(layerObject, previouLayerBackward)
            partDev = ones(1, length(previousLayerBackward));
        end
        function kaparenDeribatua = internalJakobianoaLortu(layerObject)
            kaparenDeribatua  = ones(1, length(previousLayerBackward));
        end
    end
end

