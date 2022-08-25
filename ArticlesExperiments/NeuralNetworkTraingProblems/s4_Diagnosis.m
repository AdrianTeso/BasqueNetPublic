addpath('BaskeNet');
addpath('IkasketaFuntzioak');
addpath('Layer');
addpath('Layer\ProcessingLayer');
salida=zeros(1,length(in));
error=zeros(1,length(in));
for iter= 1:length(in)
    [salida(:,iter), bn] = process(bn,in(:,iter));
end
error=tg-salida;
errorabs=abs(error);
ref=[1:length(in)];
MatrizErrorAbs=[ref;errorabs];
MatrizErrorAbs=MatrizErrorAbs';
MatrizErrorAbs=sortrows(MatrizErrorAbs,2);
Nelementos=20;
MatrizMenosErrorAbs=MatrizErrorAbs(1:Nelementos,:);
MatrizErrorAbs=MatrizErrorAbs(end-Nelementos:end,:);

RefErrorabsmasAltos=MatrizErrorAbs(:,1);
PatronesConMasError=zeros(2,Nelementos);
for iter=1:length(RefErrorabsmasAltos)
   PatronesConMasError(1,iter)=in(1, RefErrorabsmasAltos(iter));
   PatronesConMasError(2,iter)=in(2, RefErrorabsmasAltos(iter));
end
PatronesConMasError = PatronesConMasError';
PatronesConMasError = [PatronesConMasError,MatrizErrorAbs(:,2)];
aprendizaje =  AnalizaCapas(5, bn);
aprendizaje = incrementoEpoca(aprendizaje);

for iter = 1:height(PatronesConMasError)
    buscar=[PatronesConMasError(iter,1);PatronesConMasError(iter,2)];
    for iter2=1:bn.Analizador.CurrentIndex
        patr= bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter2);
        if isequal(patr.valor,buscar)
            aprendizaje.ModificadorPeso.mapa.epoca.patron(iter) = bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter2);
        end
    end
end

RefErrorabsmasBajos=MatrizMenosErrorAbs(:,1);
PatronesConMenosError=zeros(2,Nelementos);
for iter=1:length(RefErrorabsmasBajos)
   PatronesConMenosError(1,iter)=in(1, RefErrorabsmasBajos(iter));
   PatronesConMenosError(2,iter)=in(2, RefErrorabsmasBajos(iter));
end
PatronesConMenosError = PatronesConMenosError';
PatronesConMenosError = [PatronesConMenosError,MatrizMenosErrorAbs(:,2)];
aprendizajeBien =  AnalizaCapas(5, bn);
aprendizajeBien = incrementoEpoca(aprendizajeBien);

for iter = 1:height(PatronesConMenosError)
    buscar=[PatronesConMenosError(iter,1);PatronesConMenosError(iter,2)];
    for iter2=1:bn.Analizador.CurrentIndex
        patr= bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter2);
        if isequal(patr.valor,buscar)
            aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter) = bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter2);
        end
    end
end

figure('Name', 'Layer 5')
subplot(3,3,1)
mediaCapa5=zeros(1,20);
errorMedio = 0;
mediaEntrada5 = zeros(20,1);
for iter = 1:length(in)
    mediaCapa5 = mediaCapa5+bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(5).Jacobiano;
    errorMedio = errorMedio + bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Loss;
    mediaEntrada5 = mediaEntrada5 +  bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(5).LastInput;
end
mediaCapa5 = mediaCapa5 ./length(in);
errorMedio = errorMedio / length(in);
mediaEntrada5 = mediaEntrada5 ./length(in);
BackwardCapa5Medio = errorMedio*mediaCapa5;
bar(mediaCapa5)
ylabel("Avg. value")
xlabel("Synaptic weight")
title("Avg. Weight Matrix")
subplot(3,3,2)
mediaMala=zeros(1,20);
mediaBuena = zeros(1,20);
mediaErrorMala = 0;
mediaErrorBuena = 0;
for iter = 1:20
    mediaMala = mediaMala +  aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Capa(5).Jacobiano;
    mediaErrorMala = mediaErrorMala + aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Loss;
    mediaBuena = mediaBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Capa(5).Jacobiano;
    mediaErrorBuena = mediaErrorBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Loss;
end
mediaMala = mediaMala/20;
mediaBuena = mediaBuena/20;
mediaErrorMala = mediaErrorMala/20;
mediaErrorBuena = mediaErrorBuena/20;
BackwardCapa5Malo = mediaErrorMala * mediaMala;
BackwardCapa5Bueno = mediaErrorBuena * mediaBuena;
bar(mediaMala)
ylabel("Avg. value")
xlabel("Synaptic weight")
title("Avg. Worst Learning Patterns Weight Matrix")
subplot(3,3,4)
bar(dot(errorMedio,mediaErrorMala)/(norm(mediaErrorMala)*norm(errorMedio)));
ylabel("cos(\phi)")
xlabel("Synaptic weight")
title("Avg. Worst Learning Patterns partial dLoss/dX5 COS(PHI) to Total Avg.");
subplot(3,3,7)
bar(dot(errorMedio,mediaErrorBuena)/(norm(mediaErrorBuena)*norm(errorMedio)));
title("Avg. Best Learning Patterns partial dLoss/dX5 COS(PHI) to Total Avg.");
subplot(3,3,3)
dotProduct = dot(mediaCapa5, mediaMala);
normMedia= norm(mediaCapa5);
normPeor= norm(mediaMala);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
xlabel("Synaptic weight")
title("Avg. Worst Learning Patters Weight Matrix COS(PHI) to Total Avg.");
subplot(3,3,5)
bar(mediaBuena)
ylabel("Avg. value")
xlabel("Synaptic weight")
title("Avg. Best Learning Patterns Weight Matrix")
subplot(3,3,6)
dotProduct = dot(mediaCapa5, mediaBuena);
normMedia= norm(mediaCapa5);
normPeor= norm(mediaBuena);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
xlabel("Synaptic weight")
title("Avg. Best Learning Patterns Weight Matrix COS(PHI) to Total Avg.");
dotBackMala = dot(BackwardCapa5Medio, BackwardCapa5Malo);
dotBackBuena = dot(BackwardCapa5Medio, BackwardCapa5Bueno);
normBackMedia = norm(BackwardCapa5Medio);
normBackMala = norm(BackwardCapa5Malo);
normBackBuena = norm(BackwardCapa5Bueno);
cos_phiBackMala = dotBackMala /(normBackMedia*normBackMala);
cos_phiBackBuena = dotBackBuena /(normBackMedia*normBackBuena);
subplot(3,3,8)
bar(cos_phiBackMala);
ylabel("cos(\phi)")
xlabel("Pattern")
title("Avg. Worst Learning Patterns partial dLoss/dX4 COS(PHI) to Total Avg.");
subplot(3,3,9)
bar(cos_phiBackBuena);
ylabel("cos(\phi)")
xlabel("Pattern")
title("Avg. Best Learning Patterns partial dLoss/dX4 COS(PHI) to Total Avg.");

figure('Name', 'Layer 4')
subplot(3,3,1)
mediaCapa4=zeros(20,20);
for iter = 1:length(in)
    mediaCapa4 = mediaCapa4+bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(4).Jacobiano;
    
end
mediaCapa4 = mediaCapa4 ./length(in);
surf(1:20, 1:20, mediaCapa4)
title("Avg. Weight Matrix")
subplot(3,3,2)
mediaMala=zeros(20,20);
mediaBuena = zeros(20,20);
for iter = 1:20
    mediaMala = mediaMala +  aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Capa(4).Jacobiano;
    mediaBuena = mediaBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Capa(4).Jacobiano;
end
mediaMala = mediaMala/20;
mediaBuena = mediaBuena/20;
BackwardCapa4Malo = BackwardCapa5Malo * mediaMala;
BackwardCapa4Bueno = BackwardCapa5Bueno * mediaBuena;
BackwardCapa4Medio = BackwardCapa5Bueno *mediaCapa4;
surf(1:20, 1:20, mediaMala)
title("Avg. Worst Learning Patterns Weight Matrix")
subplot(3,3,3)
dotProduct = dot(mediaCapa4, mediaMala);
normMedia= norm(mediaCapa4);
normPeor= norm(mediaMala);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Worst Learning Patters Weight Matrix COS(PHI) to Total Avg.");
subplot(3,3,5)
surf(1:20, 1:20, mediaBuena)
title("Avg. Best Learning Patterns Weight Matrix")
subplot(3,3,6)
dotProduct = dot(mediaCapa4, mediaBuena);
normMedia= norm(mediaCapa4);
normPeor= norm(mediaBuena);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Best Learning Patters Weight Matrix COS(PHI) to Total Avg.");
dotBackMala = dot(BackwardCapa4Medio, BackwardCapa4Malo);
dotBackBuena = dot(BackwardCapa4Medio, BackwardCapa4Bueno);
normBackMedia = norm(BackwardCapa4Medio);
normBackMala = norm(BackwardCapa4Malo);
normBackBuena = norm(BackwardCapa4Bueno);
cos_phiBackMala = dotBackMala /(normBackMedia*normBackMala);
cos_phiBackBuena = dotBackBuena /(normBackMedia*normBackBuena);
subplot(3,3,8)
bar(cos_phiBackMala);
ylabel("cos(\phi)")
title("Avg. Worst Learning Patterns partial dLoss/dX3 COS(PHI) to Total Avg.");
subplot(3,3,9)
bar(cos_phiBackBuena);
title("Avg. Best Learning Patterns partial dLoss/dX3 COS(PHI) to Total Avg.");

figure('Name', 'Layer 3')
subplot(3,3,1)
mediaCapa3=zeros(20,100);
mediaEntrada3 = zeros(100,1);
for iter = 1:length(in)
    mediaCapa3 = mediaCapa3+bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(3).Jacobiano;
    mediaEntrada3 = mediaEntrada3 +bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(3).LastInput;
end
mediaCapa = mediaCapa3 ./length(in);
mediaEntrada3 = mediaEntrada3./length(in);
surf(1:100, 1:20, mediaCapa3)
title("Avg. Weight Matrix")
subplot(3,3,2)
mediaMala=zeros(20,100);
mediaBuena = zeros(20,100);
for iter = 1:20
    mediaMala = mediaMala +  aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Capa(3).Jacobiano;
    mediaBuena = mediaBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Capa(3).Jacobiano;
end
mediaMala = mediaMala/20;
mediaBuena = mediaBuena/20;
BackwardCapa3Malo = BackwardCapa4Malo * mediaMala;
BackwardCapa3Bueno = BackwardCapa4Bueno * mediaBuena;
BackwardCapa3Medio = BackwardCapa4Bueno *mediaCapa3;
surf(1:100, 1:20, mediaMala)
title("Avg. Worst Learning Patterns Weight Matrix")
subplot(3,3,3)
dotProduct = dot(mediaCapa3, mediaMala);
normMedia= norm(mediaCapa3);
normPeor= norm(mediaMala);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Worst Learning Patters Weight Matrix COS(PHI) to Total Avg.");
subplot(3,3,5)
surf(1:100, 1:20, mediaBuena)
title("Avg. Best Learning Patterns Weight Matrix")
subplot(3,3,6)
dotProduct = dot(mediaCapa3, mediaBuena);
normMedia= norm(mediaCapa3);
normPeor= norm(mediaBuena);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Best Learning Patters Weight Matrix COS(PHI) to Total Avg.");
dotBackMala = dot(BackwardCapa3Medio, BackwardCapa3Malo);
dotBackBuena = dot(BackwardCapa3Medio, BackwardCapa3Bueno);
normBackMedia = norm(BackwardCapa3Medio);
normBackMala = norm(BackwardCapa3Malo);
normBackBuena = norm(BackwardCapa3Bueno);
cos_phiBackMala = dotBackMala /(normBackMedia*normBackMala);
cos_phiBackBuena = dotBackBuena /(normBackMedia*normBackBuena);
subplot(3,3,8)
bar(cos_phiBackMala);
ylabel("cos(\phi)")
title("Avg. Worst Learning Patterns partial dLoss/dX2 COS(PHI) to Total Avg.");
subplot(3,3,9)
bar(cos_phiBackBuena);
ylabel("cos(\phi)")
title("Avg. Best Learning Patterns partial dLoss/dX2 COS(PHI) to Total Avg.");

figure('Name', 'Layer 2')
subplot(3,3,1)
mediaCapa2=zeros(100,100);
for iter = 1:length(in)
    mediaCapa2 = mediaCapa2+bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(2).Jacobiano;
end
mediaCapa2 = mediaCapa2 ./length(in);
surf(1:100, 1:100, mediaCapa2)
title("Avg. Weight Matrix")
subplot(3,3,2)
mediaMala=zeros(100,100);
mediaBuena = zeros(100,100);
for iter = 1:20
    mediaMala = mediaMala +  aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Capa(2).Jacobiano;
    mediaBuena = mediaBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Capa(2).Jacobiano;
end
mediaMala = mediaMala/20;
mediaBuena = mediaBuena/20;
BackwardCapa2Malo = BackwardCapa3Malo * mediaMala;
BackwardCapa2Bueno = BackwardCapa3Bueno * mediaBuena;
BackwardCapa2Medio = BackwardCapa3Bueno *mediaCapa2;
surf(1:100, 1:100, mediaMala)
title("Avg. Worst Learning Patterns Weight Matrix")
subplot(3,3,3)
dotProduct = dot(mediaCapa2, mediaMala);
normMedia= norm(mediaCapa2);
normPeor= norm(mediaMala);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Worst Learning Patters Weight Matrix COS(PHI) to Total Avg.");
subplot(3,3,5)
surf(1:100, 1:100, mediaBuena)
title("Avg. Best Learning Patterns Weight Matrix")
subplot(3,3,6)
dotProduct = dot(mediaCapa2,mediaBuena);
normMedia= norm(mediaCapa2);
normPeor= norm(mediaBuena);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Best Learning Patters Weight Matrix COS(PHI) to Total Avg.");
dotBackMala = dot(BackwardCapa2Medio, BackwardCapa2Malo);
dotBackBuena = dot(BackwardCapa2Medio, BackwardCapa2Bueno);
normBackMedia = norm(BackwardCapa2Medio);
normBackMala = norm(BackwardCapa2Malo);
normBackBuena = norm(BackwardCapa2Bueno);
cos_phiBackMala = dotBackMala /(normBackMedia*normBackMala);
cos_phiBackBuena = dotBackBuena /(normBackMedia*normBackBuena);
subplot(3,3,8)
bar(cos_phiBackMala);
ylabel("cos(\phi)")
title("Avg. Worst Learning Patterns partial dLoss/dX1 COS(PHI) to Total Avg.");
subplot(3,3,9)
bar(cos_phiBackBuena);
ylabel("cos(\phi)")
title("Avg. Best Learning Patterns partial dLoss/dX1 COS(PHI) to Total Avg.");

figure('Name', 'Layer 1')
subplot(3,3,1)
mediaCapa1=zeros(100,2);
mediaEntrada1 = [0.0;0.0];
for iter = 1:length(in)
    mediaCapa1 = mediaCapa1+bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(1).Jacobiano;
    mediaEntrada1 = mediaEntrada1 + bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(1).LastInput;
end
mediaCapa1 = mediaCapa1 ./length(in);
mediaEntrada1 = mediaEntrada1 ./length(in);
surf(1:2, 1:100, mediaCapa1)
title("Avg. Weight Matrix")
subplot(3,3,2)
mediaMala=zeros(100,2);
mediaBuena = zeros(100,2);
for iter = 1:20
    mediaMala = mediaMala +  aprendizaje.ModificadorPeso.mapa.epoca.patron(iter).Capa(1).Jacobiano;
    mediaBuena = mediaBuena + aprendizajeBien.ModificadorPeso.mapa.epoca.patron(iter).Capa(1).Jacobiano;
end
mediaMala = mediaMala/20;
mediaBuena = mediaBuena/20;
BackwardCapa1Malo = BackwardCapa2Malo * mediaMala;
BackwardCapa1Bueno = BackwardCapa2Bueno * mediaBuena;
BackwardCapa1Medio = BackwardCapa2Bueno *mediaCapa1;
surf(1:2, 1:100, mediaMala)
title("Avg. Worst Learning Patterns Weight Matrix")
subplot(3,3,3)
dotProduct = dot(mediaCapa1, mediaMala);
normMedia= norm(mediaCapa1);
normPeor= norm(mediaMala);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Worst Learning Patters Weight Matrix COS(PHI) to Total Avg.");
subplot(3,3,5)
surf(1:2, 1:100,mediaBuena)
title("Avg. Best Learning Patterns Weight Matrix")
subplot(3,3,6)
dotProduct = dot(mediaCapa1, mediaBuena);
normMedia= norm(mediaCapa1);
normPeor= norm(mediaBuena);
cos_phi= dotProduct/(normMedia * normPeor);
bar(cos_phi)
ylabel("cos(\phi)")
title("Avg. Best Learning Patters Weight Matrix COS(PHI) to Total Avg.");
dotBackMala = dot(BackwardCapa1Medio, BackwardCapa1Malo);
dotBackBuena = dot(BackwardCapa1Medio, BackwardCapa1Bueno);
normBackMedia = norm(BackwardCapa1Medio);
normBackMala = norm(BackwardCapa1Malo);
normBackBuena = norm(BackwardCapa1Bueno);
cos_phiBackMala = dotBackMala /(normBackMedia*normBackMala);
cos_phiBackBuena = dotBackBuena /(normBackMedia*normBackBuena);
subplot(3,3,8)
bar(cos_phiBackMala);
ylabel("cos(\phi)")
title("Avg. Worst Learning Patterns partial dLoss/dIn COS(PHI) to Total Avg.");
subplot(3,3,9)
bar(cos_phiBackBuena);
ylabel("cos(\phi)")
title("Avg. Best Learning Patterns partial dLoss/dIn COS(PHI) to Total Avg.");

disp("Pattern By Pattern")
fileID =  fopen('WorstLearningPatterns.txt', 'w');
PHIWorst=zeros(20,3);
MODWorst=zeros(20,3);
for index = 1:20
    tit = (strcat("Worst Case Pattern ", sprintf("%i",index), " [", sprintf("%.3f",aprendizaje.ModificadorPeso.mapa.epoca.patron(index).valor(1)), ";", sprintf("%.3f",aprendizaje.ModificadorPeso.mapa.epoca.patron(index).valor(2)), "]"));
    disp(tit)
    fprintf(fileID, tit);
    fprintf(fileID, "\n");
    a = dot(errorMedio,aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss)/(norm(errorMedio)*norm(aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX1 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss) / norm(errorMedio);
    f = strcat("Partial dLoss/dX1 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = errorMedio * mediaEntrada5;
    mediaLiso=g';
    h = aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(5).LastInput;
    paternLiso = h';
    i = dot(g,h)/(norm(g)*norm(h));
    PHIWorst(index,1)= i;
    j = norm(h)/norm(g);
    MODWorst(index,1)=j;
    k = strcat("Partial dLoss/dW_X_1 COS(PHI) to Total Avg: ", sprintf("%.2f",i), " PHI=", sprintf("%.2f",acos(i)*180/pi));
    fprintf(fileID, k);
    fprintf(fileID, "\n");
    l = strcat("Partial dLoss/dW_X_1 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    

    Back5= aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    a = (dot(BackwardCapa5Medio,Back5)/(norm(Back5)*norm(BackwardCapa5Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX2 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back5) / norm(BackwardCapa5Medio);
    f = strcat("Partial dLoss/dX2 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back4= Back5*aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    a = (dot(BackwardCapa4Medio,Back4)/(norm(Back4)*norm(BackwardCapa4Medio)));
    
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX3 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back4) / norm(BackwardCapa4Medio);
    f = strcat("Partial dLoss/dX3 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = reshape(BackwardCapa4Medio' * mediaEntrada3', 1, length(BackwardCapa4Medio) * length(mediaEntrada3));
    mediaLiso=[mediaLiso, g];
    h = reshape(Back4' * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(3).LastInput', 1, length(BackwardCapa4Medio) * length(mediaEntrada3));
    paternLiso = [paternLiso, h];
    i = dot(g,h)/(norm(g)*norm(h));
    PHIWorst(index,2)= i;
    j = norm(h)/norm(g);
    
    MODWorst(index,2)=j;
    fprintf(fileID, "Partial dLoss/dW_X_3 COS(PHI) to Total Avg:\n");
    for iter=1:length(i)
        k = strcat("\t", sprintf("%.2f",i(iter)), " PHI=", sprintf("%.2f",acos(i(iter))*180/pi),"\n");
        fprintf(fileID, k);
    end
    l = strcat("Partial dLoss/dW_X_3 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back3= Back4 *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(3).Jacobiano;
    a = (dot(BackwardCapa3Medio,Back3)/(norm(Back3)*norm(BackwardCapa3Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX4 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back3) / norm(BackwardCapa3Medio);
    f = strcat("Partial dLoss/dX4 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back2= Back3 *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(2).Jacobiano;
    a = (dot(BackwardCapa2Medio,Back2)/(norm(Back2)*norm(BackwardCapa2Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX5 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back2) / norm(BackwardCapa2Medio);
    f = strcat("Partial dLoss/dX5 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = reshape(BackwardCapa2Medio' * mediaEntrada1', 1, length(BackwardCapa2Medio) * length(mediaEntrada1));
    mediaLiso=[mediaLiso, g];
    h = reshape(Back2' * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(1).LastInput', 1, length(Back2)*length(aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(1).LastInput));
    paternLiso = [paternLiso, h];
    i = dot(g,h)/(norm(g)*norm(h));
    PHIWorst(index,3)= i;
    j = norm(h)/norm(g);
    MODWorst(index,3)=j;
    fprintf(fileID, "Partial dLoss/dW_X_5 COS(PHI) to Total Avg:\n");
    for iter=1:length(i)
        k = strcat("\t", sprintf("%.2f",i(iter)), " PHI=", sprintf("%.2f",acos(i(iter))*180/pi),"\n");
        fprintf(fileID, k);
    end
    l = strcat("Partial dLoss/dW_X_5 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back1= Back2 *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(1).Jacobiano;
    a = (dot(BackwardCapa1Medio,Back1)/(norm(Back1)*norm(BackwardCapa1Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dIn COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back1) / norm(BackwardCapa1Medio);
    f = strcat("Partial dLoss/dIn Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    cosphiLiso = dot(mediaLiso,paternLiso)/(norm(mediaLiso)*norm(paternLiso));
    relLiso = norm(paternLiso) / norm(mediaLiso);
    ListaCosPhiLisos(index) = cosphiLiso;
    ListaRelLiso(index) = relLiso;
end
figure('Name','Cos(phi) WorstPatterns United');
bar(ListaCosPhiLisos)
ylabel("cos(\phi)")
xlabel("The 20 Worst Learnt Patterns")
figure('Name','Module relationship WorstPatterns United');
bar(ListaRelLiso);
xlabel("The 20 Worst Learnt Patterns")
ylabel("Module relationship pattern to avg.")

fclose(fileID);
fileID =  fopen('BestLearningPatterns.txt', 'w');
PHIBest=zeros(20,3);
MODBest=zeros(20,3);
for index = 1:20
  	tit = (strcat("Best Case Pattern ", sprintf("%i",index), " [", sprintf("%.3f",aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).valor(1)), ";", sprintf("%.3f",aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).valor(2)), "]"));
    disp(tit)
    fprintf(fileID, tit);
    fprintf(fileID, "\n");
    a = dot(errorMedio,aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss)/(norm(errorMedio)*norm(aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX1 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss) / norm(errorMedio);
    f = strcat("Partial dLoss/dX1 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = errorMedio * mediaEntrada5;
    mediaLiso=g';
    h = aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(5).LastInput;
    paternLiso = h';
    i = dot(g,h)/(norm(g)*norm(h));
    PHIBest(index,1) = i;
    j = norm(h)/norm(g);
    MODBest(index,1) = j;
    k = strcat("Partial dLoss/dW_X_1 COS(PHI) to Total Avg: ", sprintf("%.2f",i), " PHI=", sprintf("%.2f",acos(i)*180/pi));
    fprintf(fileID, k);
    fprintf(fileID, "\n");
    l = strcat("Partial dLoss/dW_X_1 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    
    Back5= aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    a = (dot(BackwardCapa5Medio,Back5)/(norm(Back5)*norm(BackwardCapa5Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX2 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back5) / norm(BackwardCapa5Medio);
    f = strcat("Partial dLoss/dX2 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back4= Back5*aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    a = (dot(BackwardCapa4Medio,Back4)/(norm(Back4)*norm(BackwardCapa4Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX3 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back4) / norm(BackwardCapa4Medio);
    f = strcat("Partial dLoss/dX3 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = reshape(BackwardCapa4Medio' * mediaEntrada3', 1, length(BackwardCapa4Medio)*length(mediaEntrada3));
    h = reshape(Back4' * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(3).LastInput', 1, length(BackwardCapa4Medio)*length(mediaEntrada3));
    mediaLiso=[mediaLiso, g];
    paternLiso = [paternLiso, h];
    i = dot(g,h)/(norm(g)*norm(h));
    PHIBest(index,2) = i;
    j = norm(h)/norm(g);
    MODBest(index,2) = j;
    fprintf(fileID, "Partial dLoss/dW_X_3 COS(PHI) to Total Avg:\n");
    for iter=1:length(i)
        k = strcat("\t", sprintf("%.2f",i(iter)), " PHI=", sprintf("%.2f",acos(i(iter))*180/pi),"\n");
        fprintf(fileID, k);
    end
    l = strcat("Partial dLoss/dW_X_3 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back3= Back4 *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(3).Jacobiano;
    a = (dot(BackwardCapa3Medio,Back3)/(norm(Back3)*norm(BackwardCapa3Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX4 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back3) / norm(BackwardCapa3Medio);
    f = strcat("Partial dLoss/dX4 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back2= Back3 *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(2).Jacobiano;
    a = (dot(BackwardCapa2Medio,Back2)/(norm(Back2)*norm(BackwardCapa2Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dX5 COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back2) / norm(BackwardCapa2Medio);
    f = strcat("Partial dLoss/dX5 Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    g = reshape(BackwardCapa2Medio' * mediaEntrada1', 1, length(BackwardCapa2Medio)*length(mediaEntrada1));
    h = reshape(Back2' * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(1).LastInput', 1, length(BackwardCapa2Medio)*length(mediaEntrada1));
    mediaLiso=[mediaLiso, g];
    paternLiso = [paternLiso, h];
    i = dot(g,h)/(norm(g)*norm(h));
    PHIBest(index,3) = i;
    j = norm(h)/norm(g);
    MODBest(index,3) = j;
    fprintf(fileID, "Partial dLoss/dW_X_5 COS(PHI) to Total Avg:\n");
    for iter=1:length(i)
        k = strcat("\t", sprintf("%.2f",i(iter)), " PHI=", sprintf("%.2f",acos(i(iter))*180/pi),"\n");
        fprintf(fileID, k);
    end
    l = strcat("Partial dLoss/dW_X_5 Module to Total Avg. relationship: ", sprintf("%.2f",j));
    fprintf(fileID, l);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");

    Back1= Back2 *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(1).Jacobiano;
    a = (dot(BackwardCapa1Medio,Back1)/(norm(Back1)*norm(BackwardCapa1Medio)));
    arco = acos(a)*180/pi;
    b = sprintf("%.2f",a);
    d = sprintf("%.2f",arco);
    c = strcat("Partial dLoss/dIn COS(PHI) to Total Avg.: " , b, " PHI= ", d, "Degrees");
    disp(c)
    fprintf(fileID, c);
    fprintf(fileID, "\n");
    e = norm(Back1) / norm(BackwardCapa1Medio);
    f = strcat("Partial dLoss/dIn Module to Total Avg. relationship: ", sprintf("%.2f",e));
    fprintf(fileID, f);
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    fprintf(fileID, "\n");
    cosphiLiso = dot(mediaLiso,paternLiso)/(norm(mediaLiso)*norm(paternLiso));
    relLiso = norm(paternLiso) / norm(mediaLiso);
    ListaCosPhiLisos(index) = cosphiLiso;
    ListaRelLiso(index) = relLiso;
end
fclose(fileID);
figure('Name','Cos(phi) Best Patterns United');
bar(ListaCosPhiLisos)
ylabel("cos(\phi)")
xlabel("The 20 Best Learnt Patterns")
figure('Name','Module Relationship Best Patterns United');
bar(ListaRelLiso);
xlabel("The 20 Best Learnt Patterns")
ylabel("Module relationship pattern to avg.")
