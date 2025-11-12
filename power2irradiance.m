function I = power2irradiance(P,d)
%
%
% P := Power in Watts
% d := Diameter aperture in meters


Area = pi*(d/2)^2;

I = P/Area;
