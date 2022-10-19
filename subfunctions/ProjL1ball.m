function[u] = ProjL1ball(u, f, epsilon)

v = abs(u(:) - f(:));
sv = sort(v, 'descend');
cumsv = cumsum(sv);
J = 1:numel(v);
thetaCand = (cumsv - epsilon)./J';
rho = find(((sv - thetaCand)>0), 1, 'last');

theta = thetaCand(rho);
%disp(['theta: ', num2str(theta)])
v = v - theta;
v(v<0) = 0;
v((u-f)<0) = v((u-f)<0)*-1;
v = reshape(v, size(u));

u = f + v;