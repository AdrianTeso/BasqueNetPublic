close all
%%Peores Patrones
%%W_1
w1 =[];
for index=1:20
    w1 = [w1,aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(5).LastInput];
end
siuacion_medio = errorMedio * mediaEntrada5;
for index = 1:20
    patternLiso = w1(:,index)';
    cosphi(index) = dot(siuacion_medio,w1(:,index))/(norm(siuacion_medio)*norm(w1(:,index))); 
    mod(index) = norm(w1(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 1")
xlabel("The 20 Worst Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 1")
xlabel("The 20 Worst Learnt Patterns")

%%W_3
w3 =[];
for index=1:20
    Back5= aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    Back4= Back5*aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    rs = reshape(Back4' * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(3).LastInput', 1, length(BackwardCapa4Medio)*length(mediaEntrada3));
    w3 = [w3,rs'];
end
siuacion_medio = reshape(BackwardCapa4Medio' * mediaEntrada3',1,length(BackwardCapa4Medio)*length(mediaEntrada3));
for index = 1:20
    patternLiso = w3(:,index)';
    cosphi(index) = dot(siuacion_medio,w3(:,index)')/(norm(siuacion_medio)*norm(w3(:,index)')); 
    mod(index) = norm(w3(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 3")
xlabel("The 20 Worst Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 3")
xlabel("The 20 Worst Learnt Patterns")

%%W_5
w5 =[];
for index=1:20
    Back5= aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    Back4= Back5*aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    Back3= Back4*aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(3).Jacobiano;
    Back2= Back3*aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(2).Jacobiano;
    rs = reshape(Back2' * aprendizaje.ModificadorPeso.mapa.epoca.patron(index).Capa(1).LastInput', 1, length(BackwardCapa2Medio)*length(mediaEntrada1));
    w5 = [w5,rs'];
end
siuacion_medio = reshape(BackwardCapa2Medio' * mediaEntrada1',1,length(BackwardCapa2Medio)*length(mediaEntrada1));
for index = 1:20
    patternLiso = w5(:,index)';
    cosphi(index) = dot(siuacion_medio,w5(:,index)')/(norm(siuacion_medio)*norm(w5(:,index)')); 
    mod(index) = norm(w5(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 5")
xlabel("The 20 Worst Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 5")
xlabel("The 20 Worst Learnt Patterns")

%%Mejores Patrones
w1 =[];
for index=1:20
    w1 = [w1,aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(5).LastInput];
end
siuacion_medio = errorMedio * mediaEntrada5;
for index = 1:20
    patternLiso = w1(:,index)';
    cosphi(index) = dot(siuacion_medio,w1(:,index))/(norm(siuacion_medio)*norm(w1(:,index))); 
    mod(index) = norm(w1(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 1")
xlabel("The 20 Best Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 1")
xlabel("The 20 Best Learnt Patterns")

%%W_3
w3 =[];
for index=1:20
    Back5= aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    Back4= Back5*aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    rs = reshape(Back4' * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(3).LastInput', 1, length(BackwardCapa4Medio)*length(mediaEntrada3));
    w3 = [w3,rs'];
end
siuacion_medio = reshape(BackwardCapa4Medio' * mediaEntrada3',1,length(BackwardCapa4Medio)*length(mediaEntrada3));
for index = 1:20
    patternLiso = w3(:,index)';
    cosphi(index) = dot(siuacion_medio,w3(:,index)')/(norm(siuacion_medio)*norm(w3(:,index)')); 
    mod(index) = norm(w3(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 3")
xlabel("The 20 Best Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 3")
xlabel("The 20 Best Learnt Patterns")

%%W_5
w5 =[];
for index=1:20
    Back5= aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Loss *aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(5).Jacobiano;
    Back4= Back5*aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(4).Jacobiano;
    Back3= Back4*aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(3).Jacobiano;
    Back2= Back3*aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(2).Jacobiano;
    rs = reshape(Back2' * aprendizajeBien.ModificadorPeso.mapa.epoca.patron(index).Capa(1).LastInput', 1, length(BackwardCapa2Medio)*length(mediaEntrada1));
    w5 = [w5,rs'];
end
siuacion_medio = reshape(BackwardCapa2Medio' * mediaEntrada1',1,length(BackwardCapa2Medio)*length(mediaEntrada1));
for index = 1:20
    patternLiso = w5(:,index)';
    cosphi(index) = dot(siuacion_medio,w5(:,index)')/(norm(siuacion_medio)*norm(w5(:,index)')); 
    mod(index) = norm(w5(:,index)) / norm(siuacion_medio);
end
figure()
bar(cosphi)
ylabel("cos(\phi) on Layer 5")
xlabel("The 20 Best Learnt Patterns")
figure()
bar(mod)
ylabel("Module relationship pattern to avg. on Layer 5")
xlabel("The 20 Best Learnt Patterns")


