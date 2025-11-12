function p = pdf_GammaGamma_wrapped(RytovVar,I)

[alpha, beta, ~] = getNumberEddies(RytovVar);
p = pdf_GammaGamma(alpha,beta,I);