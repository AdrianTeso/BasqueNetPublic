classdef AnalizaCapas
    %ANALIZACAPAS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ModificadorPeso
        CurrentIndex
        CurrentEpoch
        regis
		NivelCorteLoss
		NivelCorteError
    end
    
    methods
        function analizarCapasObj = AnalizaCapas(numLayers,baskenetObj)
            analizarCapasObj.ModificadorPeso = struct();
            analizarCapasObj.ModificadorPeso.mapa = struct();
            analizarCapasObj.ModificadorPeso.mapa.epoca= struct();
            analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron = struct();
            analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).valor = 0;
            analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa = struct();
            analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(1) = struct();
            for iter = 1:2:numLayers
                analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(iter).modificacionPesos = baskenetObj.Layers(iter).Name.WeightsMatrix;
                analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(iter).modificacionBias = baskenetObj.Layers(iter).Name.BiasMatrix;
            end
            for iter = 1:numLayers
                if isa(baskenetObj.Layers(iter).Name,'FullyConnected')
                    analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(iter).Jacobiano = baskenetObj.Layers(iter).Name.LastWeightMatrix;
                    analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(iter).LastInput = baskenetObj.Layers(iter).Name.LastInput;
                else
                    analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Capa(iter).Jacobiano = baskenetObj.Layers(iter).Name.ProcessMatrix;
                end
                
            end
            analizarCapasObj.ModificadorPeso.mapa.epoca(1).patron(1).Loss = 0;
            analizarCapasObj.CurrentIndex = 0;
            analizarCapasObj.CurrentEpoch = 0;
            analizarCapasObj.regis = false;
			analizarCapasObj.NivelCorteLoss  = 0.5;
			analizarCapasObj.NivelCorteError = 0.5;
        end
        
        function analizarCapasObj = incrementoPatron(analizarCapasObj,patron)
            if analizarCapasObj.regis 
                analizarCapasObj.CurrentIndex = analizarCapasObj.CurrentIndex + 1;
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).valor = patron;
            end
        end
        
        function analizarCapasObj = incrementoEpoca(analizarCapasObj)
            analizarCapasObj.regis = true;
            analizarCapasObj.CurrentEpoch = analizarCapasObj.CurrentEpoch + 1;
            analizarCapasObj.CurrentIndex = 0;
        end
            
    
        function analizarCapasObj = registrar(analizarCapasObj,LayerPostua, correcionBias, correcionPesos)
            if analizarCapasObj.regis
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).Capa(LayerPostua).modificacionPesos = correcionPesos;
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).Capa(LayerPostua).modificacionBias = correcionBias;
            end
        end
        function analizarCapasObj = registrarJacobiano(analizarCapasObj,LayerPostua, Jacobiano, LastInput)
            if analizarCapasObj.regis
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).Capa(LayerPostua).Jacobiano = Jacobiano;
                if(nargin == 4)
                    analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).Capa(LayerPostua).LastInput = LastInput;
                end
            end
        end
        
        function analizarCapasObj = registrarLoss(analizarCapasObj, Loss)
            if analizarCapasObj.regis
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).Loss = Loss;
            end
        end
		
		function analizarCapasObj = registrarError(analizarCapasObj, error)
			if analizarCapasObj.regis
                analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(analizarCapasObj.CurrentIndex).error = error;
            end
		end
		
		function [analizarCapasObj, sugerirParo, sugerirCambio] = analizarEvolucionRed(analizarCapasObj)
			if analizarCapasObj.regis
				promedioLoss = mean(abs(analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(1:length(patron)).Loss));
				promedioError = mean(abs(analizarCapasObj.ModificadorPeso.mapa.epoca(analizarCapasObj.CurrentEpoch).patron(1:length(patron)).error));
				lossEsAlto = promedioLoss > NivelCorteLoss;
				errorEsAlto = promedioError > NivelCorteError;
				sugerirParo = ~lossEsAlto;
				sugerirCambio = sugerirParo && errorEsAlto;
			end	
		end
    end
end

