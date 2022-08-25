x = (-1.9:0.05:2);
x_val = (-1.9:0.07:2);
x = x ./max(x);
x_val = x_val ./max(x_val);
y = x;
y_val = x_val;
[X,Y] = meshgrid(x);
[X_val, Y_val] = meshgrid(x_val);
F =zeros(length(x));
for in1 =1:length(x)
    for index =1:length(x)
        F(in1,index)= crossIt([X(in1,index),Y(in1,index)]);
    end
end
X_rs = reshape(X,[1,length(X(:,1))*length(X(1,:))]);
Y_rs = reshape(Y,[1,length(Y(:,1))*length(Y(1,:))]);
in=[X_rs;Y_rs];
tg = reshape(F,[1,length(F(:,1))*length(F(1,:))]);
F_val =zeros(length(x_val));
for in1 =1:length(x_val)
    for index =1:length(x_val)
        F_val(in1,index)= crossIt([X_val(in1,index),Y_val(in1,index)]);
    end
end
X_rs_val = reshape(X_val,[1,length(X_val(:,1))*length(X_val(1,:))]);
Y_rs_val = reshape(Y_val,[1,length(Y_val(:,1))*length(Y_val(1,:))]);
in_val=[X_rs_val;Y_rs_val];
tg_val = reshape(F_val,[1,length(F_val(:,1))*length(F_val(1,:))]);