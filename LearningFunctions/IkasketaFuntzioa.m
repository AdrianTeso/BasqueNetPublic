classdef IkasketaFuntzioa
    %IKASKETAFUNTZIOA Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function ikasketaFuntzioObjetua = IkasketaFuntzioa()
            
        end
        
        function  error = erroreaKalkulatu(ikasfObj, irteera, esperotakoIrteera)
            error = barrukoErroreaKalkulatu(ikasfObj,irteera, esperotakoIrteera);
        end
        
        function errorDeribatua = backward(ikasfObj, irteera, esperotakoIrteera)
            errorDeribatua = barrukoErrorDeribatua(ikasfObj, irteera, esperotakoIrteera);
        end
    end
    methods(Access = protected)
         function  error = barrukoErroreaKalkulatu(ikasfObj, irteera, esperotakoIrteera)
             error = esperotakoIrteera - irteera;
         end
         function errorDeribatua = barrukoErrorDeribatua(ikasfObj, irteera, esperotakoIrteera)
             errorDeribatua = ones(1,length(irteera));
         end
    end
end

