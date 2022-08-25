for cuenta =1:length(in_val)
    [out(cuenta),bn] = process(bn, in_val(:,cuenta));
end
ou_rs = reshape(out,[length(F_val(:,1)),length(F_val(1,:))]);
figure('Name','Result');
surf(X_val,Y_val,ou_rs);
title("Output obtained from the DNN")
figure('Name', 'Target');
surf(X_val,Y_val,F_val);
title("Expected output")
error = (1/2) * (abs(F_val-ou_rs).^2);
figure('Name', 'Error');
surf(X_val,Y_val,error);
title("Resulting error");