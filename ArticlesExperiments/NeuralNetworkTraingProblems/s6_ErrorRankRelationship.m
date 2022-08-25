salida = zeros([1,length(in)]);
for iter= 1:length(in)
    [salida(:,iter), bn] = process(bn,in(:,iter));
end

error=abs(tg-salida);
error = error ./max(error);
jacob1 = struct();
jacob2 = struct();
jacob3 = struct();
jacob4 = struct();
jacob5 = struct();
for iter=1:bn.Analizador.CurrentIndex
    jacob1(iter).rango = rank(bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(1).Jacobiano);
    jacob1(iter).patron = in(:,iter);
    jacob1(iter).error = error(iter);
    jacob2(iter).rango = rank(bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(2).Jacobiano);
    jacob2(iter).patron = in(:,iter);
    jacob2(iter).error = error(iter);
    jacob3(iter).rango = rank(bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(3).Jacobiano);
    jacob3(iter).patron = in(:,iter);
    jacob3(iter).error = error(iter);
    jacob4(iter).rango = rank(bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(4).Jacobiano);
    jacob4(iter).patron = in(:,iter);
    jacob4(iter).error = error(iter);
    jacob5(iter).rango = rank(bn.Analizador.ModificadorPeso.mapa.epoca.patron(iter).Capa(5).Jacobiano);
    jacob5(iter).patron = in(:,iter);
    jacob5(iter).error = error(iter);
end

for iter=1:bn.Analizador.CurrentIndex
    jacob1(iter).rango = jacob1(iter).rango / 2;
    jacob2(iter).rango = jacob2(iter).rango / 100;
    jacob3(iter).rango = jacob3(iter).rango / 20;
    jacob4(iter).rango = jacob4(iter).rango / 20;
    jacob5(iter).rango = jacob5(iter).rango / 1;
end

matfromj1 = [jacob1(1:length(in)).rango; jacob1(1:length(in)).error; jacob1(1:length(in)).patron];
matfromj1 = matfromj1';
matfromj1 = sortrows(matfromj1,1);
matfromj2 = [jacob2(1:length(in)).rango; jacob2(1:length(in)).error; jacob2(1:length(in)).patron];
matfromj2 = matfromj2';
matfromj2 = sortrows(matfromj2,1);
matfromj3 = [jacob3(1:length(in)).rango; jacob3(1:length(in)).error; jacob3(1:length(in)).patron];
matfromj3 = matfromj3';
matfromj3 = sortrows(matfromj3,1);
matfromj4 = [jacob4(1:length(in)).rango; jacob4(1:length(in)).error; jacob4(1:length(in)).patron];
matfromj4 = matfromj4';
matfromj4 = sortrows(matfromj4,1);
matfromj5 = [jacob5(1:length(in)).rango; jacob5(1:length(in)).error; jacob5(1:length(in)).patron];
matfromj5 = matfromj5';
matfromj5 = sortrows(matfromj5,1);

refRank1=matfromj1(1,1);
index1 = 1;
refRank2=matfromj2(1,1);
index2 = 1;
refRank3=matfromj3(1,1);
index3 = 1;
refRank4=matfromj4(1,1);
index4 = 1;
refRank5=matfromj5(1,1);
index5 = 1;

for iter=1:bn.Analizador.CurrentIndex
    if(matfromj1(iter,1) ~=refRank1)
        matfromj1(index1:iter-1,5) = mean(matfromj1(index1:iter-1,2));
        matfromj1(index1:iter-1,6) = std(matfromj1(index1:iter-1,2));
        index1 = iter;
        refRank1=matfromj1(iter,1);
    end
    if(matfromj2(iter,1) ~=refRank2)
        matfromj2(index2:iter-1,5) = mean(matfromj2(index2:iter-1,2));
        matfromj2(index2:iter-1,6) = std(matfromj2(index2:iter-1,2));
        index2 = iter;
        refRank2=matfromj2(iter,1);
    end
    if(matfromj3(iter,1) ~=refRank3)
        matfromj3(index3:iter-1,5) = mean(matfromj3(index3:iter-1,2));
        matfromj3(index3:iter-1,6) = std(matfromj3(index3:iter-1,2));
        index3 = iter;
        refRank3=matfromj2(iter,1);
    end
    if(matfromj4(iter,1) ~=refRank4)
        matfromj4(index4:iter-1,5) = mean(matfromj4(index4:iter-1,2));
        matfromj4(index4:iter-1,6) = std(matfromj4(index4:iter-1,2));
        index4 = iter;
        refRank4=matfromj4(iter,1);
    end
    if(matfromj5(iter,1) ~=refRank5)
        matfromj5(index5:iter-1,5) = mean(matfromj5(index5:iter-1,2));
        matfromj5(index5:iter-1,6) = std(matfromj5(index5:iter-1,2));
        index5 = iter;
        refRank5=matfromj1(iter,1);
    end
end

matfromj1 = sortrows(matfromj1,2);
matfromj2 = sortrows(matfromj2,2);
matfromj3 = sortrows(matfromj3,2);
matfromj4 = sortrows(matfromj4,2);
matfromj5 = sortrows(matfromj5,2);

[prueba2,S2] = polyfit(matfromj2(:,2),matfromj2(:,1),9);
[prueba4,S4] = polyfit(matfromj4(:,2),matfromj4(:,1),9);
poly2= polyval(prueba2, matfromj2(:,2), S2);
poly4= polyval(prueba4, matfromj4(:,2), S4);

figure('Name', 'Error/Rank Relationship X4 Poly Reg')
plot(matfromj2(:,2),poly2,'k')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X2 Poly Reg')
plot(matfromj4(:,2),poly4,'k')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X5')
plot(matfromj1(:,2), matfromj1(:,1), 'xr')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X4')
plot(matfromj2(:,2), matfromj2(:,1), 'xr')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X3')
plot(matfromj3(:,2), matfromj3(:,1), 'xr')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X2')
plot(matfromj4(:,2), matfromj4(:,1), 'xr')
xlabel('Relative Error');
ylabel('Relative Rank');

figure('Name', 'Error/Rank Relationship X1')
plot(matfromj5(:,2), matfromj5(:,1), 'xr')
xlabel('Relative Error');
ylabel('Relative Rank');
