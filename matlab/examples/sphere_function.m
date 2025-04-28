function [bounds,y] = sphere_function(x)
y = x(1)^2+x(2)^2+x(3)^2;

%% Lower and Upper bounds (L_1, L_2, ..., L_N/ U_1, U_2, ..., U_N):

L_1=-5.12; U_1=5.12;
L_2=-5.12; U_2=5.12;
L_3=-5.12; U_3=5.12;

bounds = [L_1 U_1; L_2 U_2; L_3 U_3];