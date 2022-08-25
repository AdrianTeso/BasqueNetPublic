classdef ProcessingLayer < Layer
    %PROCESSINGLAYER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        Size
        ProcessMatrix
    end
    methods
        function processingLayerObject = ProcessingLayer(name, size)
            %PROCESSINGLAYER Construct an instance of this class
            %   Detailed explanation goes here
            processingLayerObject@Layer(name);
            processingLayerObject.Size = size;
            processingLayerObject.ProcessMatrix = zeros(size);
        end
        function processingLayerObject = resize(processingLayerObject,newSize)
            processingLayerObject.Size = newSize;
            processingLayerObject.ProcessMatrix =[];
            processingLayerObject.ProcessMatrix = zeros(newSize);
        end
    end
    methods(Access = protected)
        function partDev = internalBackward(layerObject, previouLayerBackward)
            partDev = previouLayerBackward * layerObject.ProcessMatrix;
        end
        function kaparenDeribatua = internalJakobianoaLortu(layerObject)
            kaparenDeribatua = layerObject.ProcessMatrix;
        end
    end
end

