classdef MSEIkasketa < IkasketaFuntzioa
    %MSEIKASKETA Summary of this class goes here
    %   Detailed explanation goes here
    
   methods(Access = protected)
         function  error = barrukoErroreaKalkulatu(ikasfObj,irteera, esperotakoIrteera)
             error = (1/2) * (abs(esperotakoIrteera - irteera).^2);
         end
         function errorDeribatua = barrukoErrorDeribatua(ikasfObj,irteera, esperotakoIrteera)
            errorDeribatua = (esperotakoIrteera - irteera)';
         end
    end
end


