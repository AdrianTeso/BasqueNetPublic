classdef Ikasketak
    %IKASKETAK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        alpha
        ikasketa
        epocas
        momentua
        errorekinEpokak
        minibatch
        entrenamendua
        mu
        randomizeInput
        normalizeLearning
    end
    
    methods
        function ikasketaObjetua = Ikasketak(alpha, ikasketa, epocas, momentua, errorekinEpokak, miniBatch, entrenamendua, randomizeInput, normalizeLearning)
           if(nargin < 1)
                alpha = 0.01;
            end
            if(nargin < 2)
                ikasketa = MSEIkasketa;
            end
            if(nargin < 3)
                epocas = 10;
            end
            if(nargin < 4)
                momentua = 0.02;
            end
            if(nargin < 5)
                errorekinEpokak = 10;
            end
            if(nargin < 6)
                miniBatch = 1;
            end
            if(nargin < 7)
                entrenamendua = 'SGD';
            end
            if(nargin < 8)
                randomizeInput =false;
            end
            if(nargin < 9)
                normalizeLearning = false;
            end
            ikasketaObjetua.alpha = alpha;
            ikasketaObjetua.ikasketa = ikasketa;
            ikasketaObjetua.epocas = epocas;
            ikasketaObjetua.momentua = momentua;
            ikasketaObjetua.errorekinEpokak = errorekinEpokak;
            ikasketaObjetua.minibatch = miniBatch;
            ikasketaObjetua.entrenamendua = entrenamendua;
            ikasketaObjetua.mu = 1;
            ikasketaObjetua.randomizeInput = randomizeInput;
            ikasketaObjetua.normalizeLearning = normalizeLearning;
        end
        
    end
end

