classdef EntropiaGurtzatutakoIkasketa < IkasketaFuntzioa
    %ENTROPIAGURTZATUTAKOIKASKETA Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Access = protected)
         function  error = barrukoErroreaKalkulatu(ikasfObj,irteera, esperotakoIrteera)
             error = -sum( esperotakoIrteera .* log(irteera));
         end
         function errorDeribatua = barrukoErrorDeribatua(ikasfObj,irteera, esperotakoIrteera)
             errorDeribatua = (-(esperotakoIrteera - irteera))';
         end
    end
end

