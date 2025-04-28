function [bounds,y] = hadel_function(x)
y = x(1)^2+2*x(2)^2-0.3*cos(3*pi*x(1))-0.4*cos(4*pi*x(2))+0.7;

%% Lower and Upper bounds (L_1, L_2, ..., L_N/ U_1, U_2, ..., U_N):

L_1=-2; U_1=2;
L_2=-2; U_2=2;

bounds = [L_1 U_1; L_2 U_2];