classdef BasqueNet
    %BASQUENET Summary of this class goes here
    %Estructura de red
% nombre:
% Numero de capas:
% 1 capa puede ser de cuatro tipos:
%  1.-Procesamiento: Relu, sigmoidea, lineal,tanh, signo, softmax
%  2.-Capas de dropout
%  3.-Capa de función de coste
    % Funcion de coste:
    %   3.1.-La función en si L(t,x0): Mean Square Root, Mean Absolute Error,
    %   Informacion cruzada
    %   3.2.-La derivada respecto: dot_L_x0
    %   3.3.-Implementación de algoritmos de regularizacion L1 y L2
%  4.-Fully connected: W, b xk-1=Wk*xk+bk;
%  5.-Convolucion:
%  6.-pooling max/mean:
%
% Condiciones de parada:
%  1.-Numero de epochs
%  2.-Cambios relativos de la función de coste
%  3.-Llegada a un valor suficientemente bajo de la función de coste
%  4.-Algoritmos de detección de overfitting
%  5.-Algoritmo de diagnostico de capacidad de la red de aprender: 
%  6.-Tiempo máximo de ejecución del entrenamiento
%
% Parametros de aprendizaje:
%   1.-Algoritmo de aprendizaje
%   2.-Algoritmo de dropout
%   3.-Learning ratio
%   4.-Minibatch
%   5.-Momentum
%   6.-Decaimiento del ratio de aprendizaje: Puede ser en función del
%   numero de iteraciones o del estado en el cual se encuentre el
%   entrenamiento.
%   7.-Datos de Entrenamiento, Validación y Test
    properties
        Name
        NumberOfLayers
        Layers
        StopConditions
        LearningParameters
        AllWeightSum
        AzkenekoEpokakoModuloak
        AzkenekoIteraziokoModuloak
        Moduloak
        Entrenauta
        Analizador
        MediaValoresCapa
        rowsInput
        columnInput
    end
    
    methods
        function basqueNetObject = BasqueNet(name)
            %BASQUENET Constructor base, seteamos el nombre
            basqueNetObject.Name = name;
            basqueNetObject.NumberOfLayers = 0;
            basqueNetObject.Layers = Layer.empty;
            Entrenauta = false;
            basqueNetObject.MediaValoresCapa = struct();
            basqueNetObject.MediaValoresCapa(1).Capa= struct();
            
        end
        
        function basqueNetObject = addLayer(basqueNetObject,newLayer)
            %ADDLAYER Nos permite añadir una capa a la red
            basqueNetObject.NumberOfLayers =  basqueNetObject.NumberOfLayers + 1;
            basqueNetObject.Layers(basqueNetObject.NumberOfLayers) = newLayer;
        end
        
        function basqueNetObject = removeLastLayer(basqueNetObject)
            %REMOVELASTLAYER Nos permite quitar la última capa introducida
            %delete(basqueNetObject.Layers(basqueNetObject.NumberOfLayers));
            basqueNetObject.NumberOfLayers =  basqueNetObject.NumberOfLayers - 1;
            basqueNetObject.AzkenekoEpokakoModuloak =[];
        end
        
        function basqueNetObject = setLearningParameters(basqueNetObject, alpha, ikasketa, epocas, momentua, errorekinEpokak, miniBatch, entrenamendua, randomizeInput, normalizeLearning)
            basqueNetObject.LearningParameters = Ikasketak(alpha, ikasketa, epocas, momentua, errorekinEpokak, miniBatch, entrenamendua, randomizeInput, normalizeLearning);
            
        end
        
        function [output, basqueNetObject]  = process(basqueNetObject, input)
            %PROCESS Función que tomando la entrada, la pasa por la red y
            %entrega la salida de la misma
            for iterator = 1: basqueNetObject.NumberOfLayers
                if(iterator == 1)
                    [output, basqueNetObject.Layers(iterator).Name] = ...
                        process(basqueNetObject.Layers(iterator).Name, input);
                else
                    if (iterator >= 3)&& (isa(basqueNetObject.Layers(iterator-2).Name,'FullyConnected')) && (basqueNetObject.Layers(iterator-2).Name.HaveResidual == true)
                        [output, basqueNetObject.Layers(iterator).Name] = ...
                            process(basqueNetObject.Layers(iterator).Name, [output;basqueNetObject.Layers(iterator-2).Name.LastInput] );
                    else
                        [output, basqueNetObject.Layers(iterator).Name] = ...
                            process(basqueNetObject.Layers(iterator).Name, output);
                    end
                end
            end
        end
        
        function [basqueNetObject, error, error_val] = ikasi(basqueNetObject, input, target, ploteatu, validationIn, validationOut)
            %IKASI Función que realiza el aprendizaje de la red, procesando
            %las entradas y corrigiendo los pesos sinápticos en función del
            %error obtenido.
            if(nargin < 4)
                ploteatu = false;
            end
            if(nargin < 6)
                validate = false;
            else
                validate = true;
            end
            
            for superiter=1:basqueNetObject.LearningParameters.epocas
                for iter=1:basqueNetObject.NumberOfLayers
                    basqueNetObject.MediaValoresCapa(superiter).Capa(iter).MaxCapa=0;
                    basqueNetObject.MediaValoresCapa(superiter).Capa(iter).MinCapa=0;
                    basqueNetObject.MediaValoresCapa(superiter).Capa(iter).MediaCapa=0;
                    basqueNetObject.MediaValoresCapa(superiter).Capa(iter).Rango=0;
                end
            end
            basqueNetObject.Analizador = AnalizaCapas(basqueNetObject.NumberOfLayers, basqueNetObject);
            %%basqueNetObject.Analizador =  AnalizaCapas(basqueNetObject.NumberOfLayers, basqueNetObject);
            %Realizamos esta operación para dejar estable la longitud de
            %out y error. Es como en C hacer 
            %double out[epocas][longitudEntrada]
            [basqueNetObject.rowsInput, basqueNetObject.columnInput] = size(input);
            out = zeros(basqueNetObject.LearningParameters.epocas,height(target),basqueNetObject.columnInput);
            error = zeros(basqueNetObject.LearningParameters.epocas,height(target),basqueNetObject.columnInput);
            if(validate)
                error_val = zeros(basqueNetObject.LearningParameters.epocas,length(validationIn));
            else
                error_val = zeros(basqueNetObject.LearningParameters.epocas,1);
            end
            f = waitbar(0,'Ikasketa hazi egin da...', 'Name', 'Datuetatik ikasten');
            cuentaError = 0;
            figure('Name', 'Ikasketa datuak')
            X = (1:basqueNetObject.NumberOfLayers);
            Y = (1:basqueNetObject.columnInput);
            for cuentaEpocas = 1:basqueNetObject.LearningParameters.epocas
                tic()
                if cuentaEpocas == basqueNetObject.LearningParameters.epocas
                    basqueNetObject.Analizador = incrementoEpoca(basqueNetObject.Analizador);
                end
                [basqueNetObject, out(cuentaEpocas,:,:), error(cuentaEpocas,:,:)] =...
                    epokaBatIkasi(basqueNetObject, input, target);
                
                if(validate)
                    error_val(cuentaEpocas,:) = epokaBatBalidatu(basqueNetObject,validationIn, validationOut);
                    if(abs(mean(error_val(cuentaEpocas,:))) > abs(mean(error(cuentaEpocas,:))) && mean(error_val(cuentaEpocas,:))/mean(error(cuentaEpocas,:)) > 10)
                        cuentaError = cuentaError +1;
                    else
                        cuentaError = 0;
                    end
                    if cuentaError == basqueNetObject.LearningParameters.errorekinEpokak
                        break;
                    end
                end
                if(ploteatu)
                    internalPlotear(basqueNetObject, out, error, target, cuentaEpocas);
                end
                
                tiempo = toc();
                waitbar(cuentaEpocas/basqueNetObject.LearningParameters.epocas,f,sprintf('Epoka:%i / %i Denbora Epokako: %12.6f s\nErrorea %12.9f BalidazioErrorea %12.9f',...
                                                                                          cuentaEpocas, ...
                                                                                          basqueNetObject.LearningParameters.epocas,...
                                                                                          tiempo,mean(error(cuentaEpocas,:)), mean(error_val(cuentaEpocas,:))));
                %%basqueNetObject.Moduloak(cuentaEpocas) = basqueNetObject.AzkenekoEpokakoModuloak;
                capasEnPolares(basqueNetObject, X, Y);
                for iter=1:basqueNetObject.NumberOfLayers
                    if(isa(basqueNetObject.Layers(iter).Name,'FullyConnected'))
                        basqueNetObject.MediaValoresCapa(cuentaEpocas).Capa(iter).MaxCapa = max(max(basqueNetObject.Layers(iter).Name.WeightsMatrix));
                        basqueNetObject.MediaValoresCapa(cuentaEpocas).Capa(iter).MinCapa = min(min(basqueNetObject.Layers(iter).Name.WeightsMatrix));
                        basqueNetObject.MediaValoresCapa(cuentaEpocas).Capa(iter).MediaCapa = mean(mean(basqueNetObject.Layers(iter).Name.WeightsMatrix));
                        basqueNetObject.MediaValoresCapa(cuentaEpocas).Capa(iter).Rango = range(basqueNetObject.Layers(iter).Name.WeightsMatrix, 'all');
                    end
                end  
            end
            %%if(ploteatu == false)
            %%    internalPlotear(basqueNetObject, out, error, target, cuentaEpocas);
            %%end
            Entrenauta = true;
        end
        function JacobianoXnWC = kalkulatuJakobianoBektorea(basqueNetObject, kapaZenbakia)
            
            JacobianoXNXC = kalkulatuJakobianoX(basqueNetObject, kapaZenbakia);
            
            JacobianoXnWC = JacobianoXNXC' * basqueNetObject.Layers(kapaZenbakia).Name.LastInput';
        end
        
        function JacobianoXnCBias = kalkulatuJakobianoBektoreaBias(basqueNetObject, kapaZenbakia)
            
            JacobianoXNXC = kalkulatuJakobianoX(basqueNetObject, kapaZenbakia);
            
            JacobianoXnCBias = JacobianoXNXC;
        end
        function basqueNetObject = insertarPenultimo(basqueNetObject, fcl, pl)
            lastLayer = basqueNetObject.Layers(basqueNetObject.NumberOfLayers);
            basqueNetObject.Layers(basqueNetObject.NumberOfLayers) = fcl;
            basqueNetObject.Layers(basqueNetObject.NumberOfLayers+1) = pl;
            basqueNetObject.Layers(basqueNetObject.NumberOfLayers+2) = lastLayer;
            basqueNetObject.NumberOfLayers = basqueNetObject.NumberOfLayers+2;
            basqueNetObject.AzkenekoEpokakoModuloak =[];
        end
    end
    methods(Access = protected)
        function[basqueNetObject, out, error] = epokaBatIkasi(basqueNetObject, input, target)
            %EPOKABATIKASI Hace los pasos necesarios para aprender una
            %epoca completa.
            out = zeros(height(target),basqueNetObject.columnInput);
            error = zeros(height(target),basqueNetObject.columnInput);
            if(basqueNetObject.LearningParameters.randomizeInput)
                macroVector = [double(input);target];
                longitudMacroVector= size(macroVector);
                macroVector = macroVector(:,randperm(longitudMacroVector(2)));
                inputHeigh = height(input);
                targetHeigh = height(target);
                input=[];
                for iterador = 1:inputHeigh
                    input=[input;macroVector(iterador,:)];
                end
                target=[];
                for iterador = 1:targetHeigh
                    target=[target;macroVector(iterador+inputHeigh,:)];
                end
            end
            for iterator =1 : 1 : basqueNetObject.NumberOfLayers
                if(isa(basqueNetObject.Layers(iterator).Name,'FullyConnected'))
                   basqueNetObject.Layers(iterator).Name = eraseLastModification(basqueNetObject.Layers(iterator).Name);
                end
            end
            for iterador = 1: basqueNetObject.columnInput
                    basqueNetObject.Analizador = incrementoPatron(basqueNetObject.Analizador, input(:,iterador));
                    [out(:,iterador), basqueNetObject] = ...
                        process(basqueNetObject,input(:,iterador));
                    for iterator =1 : 1 : basqueNetObject.NumberOfLayers
                        if(isa(basqueNetObject.Layers(iterator).Name,'FullyConnected'))
                            basqueNetObject.Analizador = registrarJacobiano(basqueNetObject.Analizador,iterator, basqueNetObject.Layers(iterator).Name.LastWeightMatrix,  basqueNetObject.Layers(iterator).Name.LastInput);
                        else
                            basqueNetObject.Analizador = registrarJacobiano(basqueNetObject.Analizador,iterator, basqueNetObject.Layers(iterator).Name.ProcessMatrix);
                        end
                    end
                    error(:,iterador) = ...
                        erroreaKalkulatu(basqueNetObject.LearningParameters.ikasketa,...
                                         out(:,iterador),...
                                         target(:,iterador));
                    
                    if(mod(iterador,basqueNetObject.LearningParameters.minibatch) == 0)
                        basqueNetObject = ...
                            azkenekoErroreazIkasi(basqueNetObject,...
                                                  out(:,iterador),...
                                                  target(:,iterador), true, basqueNetObject.LearningParameters.minibatch);
                    else if(iterador == basqueNetObject.columnInput)
                            basqueNetObject = ...
                            azkenekoErroreazIkasi(basqueNetObject,...
                                                  out(:,iterador),...
                                                  target(:,iterador), true, mod(iterador,basqueNetObject.LearningParameters.minibatch));
                        else
                            basqueNetObject = ...
                            azkenekoErroreazIkasi(basqueNetObject,...
                                                  out(:,iterador),...
                                                  target(:,iterador), false, mod(iterador,basqueNetObject.LearningParameters.minibatch));
                        end
                    end 
                    basqueNetObject.AzkenekoEpokakoModuloak(iterador,:) = basqueNetObject.AzkenekoIteraziokoModuloak;
            end
            for iterador = 1:basqueNetObject.NumberOfLayers
                if(isa(basqueNetObject.Layers(iterador).Name,'FullyConnected'))
                    basqueNetObject.AllWeightSum = basqueNetObject.AllWeightSum +...
                                                   basqueNetObject.Layers(iterador).Name.WeightSum;
                end
            end
        end
        function basqueNetObject = azkenekoErroreazIkasi(basqueNetObject,output, target, aplicarMinibatch, elementosMiniBatch)
            %AZKENEKOERROREAZIKASI Función que realiza un único paso de
            %aprendizaje para una entrada concreta, con una salida
            %concreta.
            
             errorDeribatua = backward(basqueNetObject.LearningParameters.ikasketa, output, target);
             basqueNetObject.Analizador = registrarLoss(basqueNetObject.Analizador,errorDeribatua);
             partDev = errorDeribatua;
             if basqueNetObject.LearningParameters.normalizeLearning 
                partDev = partDev / (norm(partDev) + realmin);
             end
             for iterator = basqueNetObject.NumberOfLayers : -1:1
                if(isa(basqueNetObject.Layers(iterator).Name,'FullyConnected'))
                    basqueNetObject = pisuakBerriztu(basqueNetObject,errorDeribatua,partDev, iterator, aplicarMinibatch, elementosMiniBatch);
                end
                partDev = backward(basqueNetObject.Layers(iterator).Name, partDev);
                if basqueNetObject.LearningParameters.normalizeLearning 
                    partDev = partDev / (norm(partDev) + realmin);
                end
                if (iterator >= 3)&& isa(basqueNetObject.Layers(iterator-2).Name,'FullyConnected') && basqueNetObject.Layers(iterator-2).Name.HaveResidual == true
                    for cuentaVector = basqueNetObject.Layers(iterator).Name.InputSize: -1: basqueNetObject.Layers(iterator-1).Name.Size+1
                        partDev(:,cuentaVector) = [];
                    end
                end
                basqueNetObject.AzkenekoIteraziokoModuloak(iterator) = sacarModulo(basqueNetObject,partDev);
             end
             basqueNetObject.LearningParameters.mu = basqueNetObject.LearningParameters.mu /1.000001;
        end
        function error_val = epokaBatBalidatu(basqueNetObject,validationIn, validationOut)
            out = zeros(1,length(validationIn));
            error_val = zeros(1,length(validationIn));
            for iterador = 1:length(validationIn)
                    [out(iterador), basqueNetObject] = ...
                        process(basqueNetObject,validationIn(:,iterador));
                    error_val(iterador) = ...
                        erroreaKalkulatu(basqueNetObject.LearningParameters.ikasketa,...
                                         out(iterador),...
                                         validationOut(:,iterador));
            end
        end
        function internalPlotear(basqueNetObject, out,error, target, epoca)
            %INTERNALPLOTEAR Permite hacer un plot de la última epoca
            %procesada.
            figure('Name','Epoca: '+string(epoca));
            subplot(2,1,1);
            plot((1:length(target)),out(epoca, :),(1:length(target)),target)
            legend('Irteera','Objetiboa');
            title('Irteera')
            subplot(2,1,2);
            plot(error(epoca, :))
            title('Errorea')
            pause(3)
        end
        function basqueNetObject = pisuakBerriztu(basqueNetObject,errorDeribatua, partDev , LayerPostua, aplicarMiniBatch, elementosMiniBatch)
            %PISUAKBERRIZTU Permite modificar los pesos sinápticos de una
            %capa fullyconnected.
            if (strcmp(basqueNetObject.LearningParameters.entrenamendua,'SGD'))
                [correccionPesos, correccionBias] = stocastic(basqueNetObject, partDev, LayerPostua);
            else
                if (strcmp(basqueNetObject.LearningParameters.entrenamendua,'Levenberg-Marquard'))
                    [correccionPesos, correccionBias] = levenbergMarquard(basqueNetObject, LayerPostua, errorDeribatua);
                else
                    [correccionPesos, correccionBias] = stocastic(basqueNetObject, partDev);
                end
                
            end
            basqueNetObject.Analizador = registrar(basqueNetObject.Analizador,LayerPostua, correccionBias, correccionPesos);
            
            correccionPesos = dropOutBN(basqueNetObject, correccionPesos,...
                    basqueNetObject.Layers(LayerPostua).Name.DropOutRate);
            correccionBias = dropOutBN(basqueNetObject, correccionBias, ...
                    basqueNetObject.Layers(LayerPostua).Name.DropOutRate);
            if size(basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos) ~= size(correccionPesos)
                basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos = reshape(basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos,basqueNetObject.Layers(LayerPostua).Name.InputSize * basqueNetObject.Layers(LayerPostua).Name.OutputSize,1);
            end
            basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos = basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos + correccionPesos;
            basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaBias = basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaBias + correccionBias;
            if aplicarMiniBatch
               
                correccionPesos = basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos ./elementosMiniBatch;
                correccionBias =  basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaBias ./elementosMiniBatch;
                if(strcmp(basqueNetObject.LearningParameters.entrenamendua,'Levenberg-Marquard')) 
                    correccionPesos = reshape(correccionPesos, basqueNetObject.Layers(LayerPostua).Name.OutputSize, basqueNetObject.Layers(LayerPostua).Name.InputSize);
                end
                momentoPesos = correccionPesos + basqueNetObject.LearningParameters.momentua * ...
                      basqueNetObject.Layers(LayerPostua).Name.LastMomentum;
               ikastekoPisuak = momentoPesos;
               
                basqueNetObject.Layers(LayerPostua).Name.LastModification = basqueNetObject.Layers(LayerPostua).Name.LastModification +abs(ikastekoPisuak);
                basqueNetObject.Layers(LayerPostua).Name.WeightsMatrix =  ...
                             basqueNetObject.Layers(LayerPostua).Name.WeightsMatrix + ikastekoPisuak;

                basqueNetObject.Layers(LayerPostua).Name.LastMomentum = momentoPesos;
             
                momentoBias = correccionBias + basqueNetObject.LearningParameters.momentua * ...
                      basqueNetObject.Layers(LayerPostua).Name.LastMomentumBias;
                ikastekoBias = momentoBias;
                
                basqueNetObject.Layers(LayerPostua).Name.BiasMatrix = ...
                     basqueNetObject.Layers(LayerPostua).Name.BiasMatrix + ikastekoBias;
                basqueNetObject.Layers(LayerPostua).Name.LastModificationBias = basqueNetObject.Layers(LayerPostua).Name.LastModificationBias + abs(ikastekoBias);
                basqueNetObject.Layers(LayerPostua).Name.LastMomentumBias = momentoBias;
                basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaPesos = zeros(basqueNetObject.Layers(LayerPostua).Name.OutputSize,basqueNetObject.Layers(LayerPostua).Name.InputSize);
                basqueNetObject.Layers(LayerPostua).Name.GuardadoMediaBias = zeros(basqueNetObject.Layers(LayerPostua).Name.OutputSize,1);
            end   
        end
        function ikastekoPisuak = dropOutBN(basqueNetObject, momentoPesos, dropOutRate)
            %DROPOUTBN Crea la matriz de aprendizaje tras DropOut
            [filas, columnas] = size(momentoPesos);
            matrizDrop = zeros(filas, columnas);
            numCandidatos = round(filas*columnas*(1-dropOutRate));
            candidatosActivos = randperm(filas * columnas, numCandidatos);
            matrizDrop(candidatosActivos) = 1/(1-dropOutRate);
            ikastekoPisuak = momentoPesos .* matrizDrop;
        end
        function moduloPesos = sacarModulo(basqueNetObject,partDev)
            moduloPesos = norm(partDev);
        end
        
        function [correccionPesos, correccionBias] = stocastic(basqueNetObject, errorDeribatua, LayerPostua)
            correccionPesos = basqueNetObject.LearningParameters.alpha * ...
                         errorDeribatua' * ...
                         basqueNetObject.Layers(LayerPostua).Name.LastInput';
            correccionBias = basqueNetObject.LearningParameters.alpha * errorDeribatua';
            
        end
        
        function [correccionPesos, correccionBias] = levenbergMarquard(basqueNetObject, LayerPostua, errorDeribatua)
            alpha = 10;
            J = kalkulatuJakobianoBektorea(basqueNetObject, LayerPostua);
            Jfila = reshape(J, 1,basqueNetObject.Layers(LayerPostua).Name.InputSize * basqueNetObject.Layers(LayerPostua).Name.OutputSize);
            JBias = kalkulatuJakobianoBektoreaBias(basqueNetObject, LayerPostua);
            
            traspuesta = Jfila';
            correccionPesos = (traspuesta * Jfila + alpha * eye(length(Jfila))) \ (traspuesta *(errorDeribatua));
            traspuesta = JBias';
            correccionBias = (traspuesta * JBias + alpha * eye(length(JBias))) \ (traspuesta * errorDeribatua);
            %%correccionPesos = reshape(correccionPesos, basqueNetObject.Layers(LayerPostua).Name.OutputSize, basqueNetObject.Layers(LayerPostua).Name.InputSize);
            
        end
        
        function capasEnPolares(basqueNetObject, X, Y)
            capas = contarFullyConnected(basqueNetObject);
            subplot(capas, 3,1)
            surf(X,Y, basqueNetObject.AzkenekoEpokakoModuloak);
            title('Deribatu partzialen moduluak');
            fullyLayer = 0;
            for iterator =1 : 1 : basqueNetObject.NumberOfLayers
                if(isa(basqueNetObject.Layers(iterator).Name,'FullyConnected'))
                    fullyLayer = fullyLayer +1;
                    subplot(capas, 3, fullyLayer * 2 + fullyLayer - 1);
                    stri = strcat(string(iterator), '. Geruzako Pisuen modifikazio akumul.');
                    if basqueNetObject.Layers(iterator).Name.InputSize > 1
                        X = 1:1:basqueNetObject.Layers(iterator).Name.InputSize;
                    else
                        X = 1;
                    end
                    if basqueNetObject.Layers(iterator).Name.OutputSize > 1
                        Y = 1: 1: basqueNetObject.Layers(iterator).Name.OutputSize;
                    else
                        Y = 1;
                    end
                    if basqueNetObject.Layers(iterator).Name.InputSize > 1 && basqueNetObject.Layers(iterator).Name.OutputSize > 1
                        surf(X, Y,basqueNetObject.Layers(iterator).Name.LastModification);
                    else
                        if basqueNetObject.Layers(iterator).Name.InputSize == 1
                            plot(basqueNetObject.Layers(iterator).Name.LastModification','b*');
                        else
                            plot(basqueNetObject.Layers(iterator).Name.LastModification,'b*');
                        end
                    end
                    title(stri);
                    stri = strcat(string(iterator), '. Geruzako Biasen modifikazio akumul.');
                    subplot(capas, 3, fullyLayer * 2 + fullyLayer);
                    plot(basqueNetObject.Layers(iterator).Name.LastModificationBias, 'b*');
                    title(stri);
                end
            end
        end
        function capas = contarFullyConnected(basqueNetObject)
            capas = 0;
            for iterator = basqueNetObject.NumberOfLayers : -1:1
                if(isa(basqueNetObject.Layers(iterator).Name,'FullyConnected'))
                    capas =  capas +1;
                end
            end
        end

        function JacobianoX = kalkulatuJakobianoX(basqueNetObject, kapaZenbakia)
            if kapaZenbakia == basqueNetObject.NumberOfLayers
                JacobianoX = 1;
            end
            for iterador = basqueNetObject.NumberOfLayers: -1: kapaZenbakia + 1
                if iterador == basqueNetObject.NumberOfLayers
                    JacobianoX = jakobianosLortu(basqueNetObject.Layers(iterador).Name);
                else
                    if iterador ~= 1
                        if isa(basqueNetObject.Layers(iterador-1).Name,'FullyConnected')
                            if basqueNetObject.Layers(iterador-1).Name.HaveResidual
                                JacobianoX = JacobianoX(1:length(JacobianoX)-basqueNetObject.Layers(iterador-1).Name.InputSize);
                            end
                        end
                    end
                    JacobianoX = JacobianoX * jakobianosLortu(basqueNetObject.Layers(iterador).Name);
                end
            end
        end
        
    end
end

