function [bounds,y] = eason_function(x)
y = -cos(x(1))*cos(x(2))*exp(-((x(1)-pi)^2+(x(2)-pi)^2));

%% Lower and Upper bounds (L_1, L_2, ..., L_N/ U_1, U_2, ..., U_N):

L_1=-50; U_1=50;
L_2=-50; U_2=50;

bounds = [L_1 U_1; L_2 U_2];