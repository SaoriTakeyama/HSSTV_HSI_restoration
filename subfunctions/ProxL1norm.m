function[out] = ProxL1norm(x, gamma, varargin)

if ~iscell(x) % array case
    if isempty(varargin) % no weight case
        x(abs(x) <= gamma) = 0;
        x(abs(x) > gamma) = x(abs(x) > gamma) - sign(x(abs(x) > gamma))*gamma;
        out = x;
    else % weighted L1 case
        w = varargin{1}; % weight
        gamma = gamma*w;
        x(abs(x) <= gamma) = 0;
        x(abs(x) > gamma) = x(abs(x) > gamma) - sign(x(abs(x) > gamma)).*gamma(abs(x) > gamma);
        out = x;
    end
else
end