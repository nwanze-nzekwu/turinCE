function p = pdf_GammaGamma(alpha,beta,I)
%function p = pdf_GammaGamma(alpha,beta,I)
%
%   Calculates the probability density function (pdf) of the gamma-gamma
%   model. Model proposed by M. A. Al-Habash in ''Mathematical model for 
%   the irradiance probability density function of a laser beam propagating
%   through turbulent media''
%
%   Inputs : 
%       alpha := number of large-scale eddies
%       beta  := number of small-scale eddies
%       I     := normalized received irradiance
%
%   Output :
%       p     := Gamma-Gamma pdf of I values
%
%   Authors: Marco Fernandes <marcofernandes@av.it.pt>
%   Last update: 31/05/2020
%   

k = (alpha+beta)/2;
k1 = alpha*beta;
K =2*(k1^k)/(gamma(alpha)*gamma(beta));
K1 = I.^(k-1);
Z = 2*sqrt(k1*I);
p = K.*K1.*besselk((alpha-beta),Z);