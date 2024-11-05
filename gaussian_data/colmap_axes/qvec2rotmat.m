function [R] = qvec2rotmat(qvec)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
R = [1 - 2*qvec(3)^2-2*qvec(4)^2,2 * qvec(2) * qvec(3) - 2 * qvec(1) * qvec(4),2 * qvec(4) * qvec(2) + 2 * qvec(1) * qvec(3);...
2 * qvec(2) * qvec(3) + 2 * qvec(1) * qvec(4), 1 - 2 * qvec(2)^2 - 2 * qvec(4)^2,2 * qvec(3) * qvec(4) - 2 * qvec(1) * qvec(2);...
2 * qvec(4) * qvec(2) - 2 * qvec(1) * qvec(3),2 * qvec(3) * qvec(4) + 2 * qvec(1) * qvec(2),1 - 2 * qvec(2)^2 - 2 * qvec(3)^2];
end