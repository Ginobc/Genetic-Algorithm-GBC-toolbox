function [bounds,y] = simple_function(x)
y = 2*x(1)^2+2*x(1)*x(2)+2*x(2)^2-6*x(1);

%% Lower and Upper bounds (L_1, L_2, ..., L_N/ U_1, U_2, ..., U_N):

L_1=-2; U_1=2;
L_2=-2; U_2=2;

bounds = [L_1 U_1; L_2 U_2];