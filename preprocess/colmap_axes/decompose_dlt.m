function [R,K,X0] = decompose_dlt(pm)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

coefs = reshape(pm',[1,12]);

H=[coefs(1),coefs(2),coefs(3);coefs(5),coefs(6),coefs(7);coefs(9),coefs(10),coefs(11)];
h=[coefs(4);coefs(8);coefs(12)];
Rz = [-1,0,0;0,-1,0;0,0,1];
X0 = -inv(H)*h;
[R,K] = qr(inv(H));
K=inv(K);
K = K*Rz/K(3,3);
R = (Rz*R')

end