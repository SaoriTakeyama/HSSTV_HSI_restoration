% Author: Shunsuke Ono (ono@isl.titech.ac.jp)

function[Du] = ProxTVnorm_Channelwise(Du, gamma)

[v, h, c, d] = size(Du);
onemat = ones(v, h, c);
thresh = ((sqrt(sum(Du.^2, 4))).^(-1))*gamma;
thresh(thresh > 1) = 1;
coef = (onemat - thresh);
Du = repmat(coef,1,1,1,d).*Du;











