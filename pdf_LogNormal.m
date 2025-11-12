function p = pdf_LogNormal(RytovVar,I)
%
%   Authors: Marco Fernandes <marcofernandes@av.it.pt>
%   Last update: 05/11/2020
  
El = -RytovVar./2;
out_exp = (1./(sqrt(2.*pi.*RytovVar)) .* 1./I);
num_exp = -(log(I)-El).^2;
den_exp = 2.*RytovVar;


p = out_exp.*exp(num_exp./den_exp);
% p=(1./(sqrt(2*pi*RytovVar))).*exp((-(log(I./1)-(-RytovVar/2)).^2)./(2*RytovVar))

% disp(['RV: ', num2str(RytovVar)])

