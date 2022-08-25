function [target,Datos, NimagenesReal]=CrearDatos(ImagenBase,Nimagenes, TamanoEnX,TamanoEnY)
    Nxmax=size(ImagenBase,2);
    Nymax=size(ImagenBase,1);
    Datos=[];
    target=[];
    %paso = ((Nxmax-TamanoEnX)*(Nymax-TamanoEnY))/Nimagenes;
    pasoX = sqrt(Nimagenes*(Nxmax-TamanoEnX)/(Nymax-TamanoEnY));
    pasoX = (Nxmax-TamanoEnX)/pasoX;
    pasoY = sqrt(Nimagenes*(Nymax-TamanoEnY)/(Nxmax-TamanoEnX));
    pasoY= (Nymax-TamanoEnY)/pasoY;
    KX = 1:pasoX:Nxmax-TamanoEnX;
    KY = 1:pasoY:Nymax-TamanoEnY;
    %% for iterador=1:Nimagenes
    for iter=1:length(KX)
        kx= KX(iter);
        for itery=1:length(KY)
        %kx=random('unid',Nxmax-TamanoEnX,1,1);
        %ky=random('unid',Nymax-TamanoEnY,1,1);
            ky = KY(itery);
            sbImagen=ImagenBase(ky:ky+TamanoEnY-1,kx:kx+TamanoEnX-1,:); %Lo convertimos en escala de grises
            Datos=[Datos,sbImagen(:)];
            target=[target,[(kx+(TamanoEnX/2))/Nxmax;(ky+(TamanoEnY/2))/Nymax]];
       end
    end
    NimagenesReal = size(target,2);
end