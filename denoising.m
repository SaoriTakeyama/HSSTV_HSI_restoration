%%%% HS denoising (gaussian) using HSSTV %%%%
% Author: Saori Takeyama (takeyama.s.aa@m.titech.ac.jp)
% Last version: July 30, 2020

function [psnr, ssim] = denoising(sigma, s_p, l_v, l_h, omega, HSI, l)
%% load rondom seed %%
load RANDOMSTATES
rand('state', rand_state);
randn('state', randn_state);

%% observation generation %%
u_org = HSI/max(abs(HSI(:)));
[rows, cols, bands] = size(u_org);
gamma = 0.1; % stepsize of ADMM

% degradation operator (in this code, we set Phi = I)
Phi = @(z) z;
Phit = @(z) z;

v = Phi(u_org) + sigma*randn(rows, cols, bands);
Sp = 0.5*ones(rows,cols,bands);
Sp = imnoise(Sp,'salt & pepper',s_p);
% line noise
hst_ind = randi(rows*bands,[int16(rows*bands*l_h),1]);
vst_ind = randi(cols*bands,[int16(cols*bands*l_v),1]);
hst = zeros(rows,bands);
hst(hst_ind) = 1;
vst = zeros(cols,bands);
vst(vst_ind) = 1;
hst = reshape(hst,rows,1,bands);
hst = repmat(hst,1,cols,1);
vst = reshape(vst,1,cols,bands);
vst = repmat(vst,rows,1,1);
st = vst + hst;
st(st>1) = 1;
Sp(st==1) = 0;
v(Sp==0) = 0;
v(Sp==1) = 1;

N = rows*cols*bands;
v_ave = sum(v(:))/N;
epsilon = 0.83 * sqrt(N * (1-(s_p*(1-l_h-l_v) + l_v + l_h - l_h * l_v)) * sigma^2); % radius of L2-ball
eta = N*(0.5*s_p*(1-l_v-l_h) + v_ave*(l_v + l_h - l_v*l_h)); % radius of L1-ball

%% setting %%

% difference operator with circulant boundary
Db = @(z) z(:,:,[2:bands, 1])-z; % difference between neighboring bands
Dbt = @(z) z(:,:,[bands,1:bands-1]) - z; % transpose of Db
Dv = @(z) z([2:rows, 1],:,:) - z; % vertical difference between neighboring pixel
Dvt = @(z) z([rows,1:rows-1],:,:) - z; % transpose of Dv
Dh = @(z) z(:,[2:cols, 1],:)-z; % horizontal difference between neighboring pixel
Dht = @(z) z(:,[cols,1:cols-1],:) - z; % transpose of Dh
D = @(z) cat(4, Dv(Db(z)), Dh(Db(z)), omega * Dv(z), omega * Dh(z)); % HSSTV
Dt = @(z) Dbt(Dvt(z(:,:,:,1))) + Dbt(Dht(z(:,:,:,2))) + omega * Dvt(z(:,:,:,3)) + omega * Dht(z(:,:,:,4));


% for inversion in update of u (F is nDFFT, D is difference operator with circulant boundary, and DtD = -L)
DbtDbFinv = zeros(rows,cols,bands);
DhtDhFinv = zeros(rows,cols,bands);
DvtDvFinv = zeros(rows,cols,bands);
DbtDbFinv(1,1,1) = 2;
DbtDbFinv(1,1,2) = -1;
DbtDbFinv(1,1,bands) = -1;
FDbtDbFinv = fftn(DbtDbFinv); % kernel of Db^tDb
DhtDhFinv(1,1,1) = 2;
DhtDhFinv(1,2,1) = -1;
DhtDhFinv(1,cols,1) = -1;
FDhtDhFinv = fftn(DhtDhFinv); % kernel of Dx^tDx
DvtDvFinv(1,1,1) = 2;
DvtDvFinv(2,1,1) = -1;
DvtDvFinv(rows,1,1) = -1;
FDvtDvFinv = fftn(DvtDvFinv); % kernel of Dy^tDy
K = FDbtDbFinv.*(FDvtDvFinv+FDhtDhFinv)+ (omega^2)*(FDvtDvFinv+FDhtDhFinv) ...
    + 3*ones(rows,cols,bands)/2;

% variables
u = v;
z1 = D(v);
z2 = v;
z3 = v;
z4 = v;
d1 = z1;
d2 = z2;
d3 = z3;
d4 = z4;

%% main loop%%
maxIter = 10000;
stopcri = 0.01;
disprate = 1;
for i = 1:maxIter
    
    upre = u;
    % update of u
    rhs = Dt(z1-d1) + Phit(z2-d2)/2 - Phit(z3-d3)/2 + z4-d4;
    u = (ifftn(fftn(rhs)./K)); % inversion via diagonalization by 2DFFT
    s = (z2 - d2 - Phi(u) + z3 - d3)/2;
    
    % update of z
    % prox of L1 or mixed L1,2 norm
    z1 = D(u) + d1;
    switch l
        case 'L1'
            z1 = ProxL1norm(z1, gamma); % p = 1, prox of L1 norm
        case 'L1,2'
            z1 = ProxTVnorm_Channelwise(z1, gamma); % p = 2, prox of mixed L1,2 norm
    end
    % prox of mixed L2 norm
    z2 = Phi(u) + s + d2;
    z2 = ProjL2ball(z2, v, epsilon);
    %proj onto L1 norm ball
    z3 = s + d3;
    z3 = ProjL1ball(z3, 0, eta);
    % proj onto dynamic range of u
    z4 = u + d4;
    z4(z4 > 1) = 1;
    z4(z4 < 0) = 0;
    
    % update of d
    d1 = d1 + D(u) - z1;
    d2 = d2 + Phi(u) + s - z2;
    d3 = d3 + s - z3;
    d4 = d4 + u - z4;
    
    % stopping condition
    res = u - upre;
    error = norm(res(:),2);
    if error < stopcri
        break;
    end
    
    % show PSNR and SSIM every disprate 
    if rem(i, disprate) == 0
        psnr = EvalImgQuality(u, u_org, 'PSNR');
        ind = u>0;
        ssim = ssim_index3d(255*u,255*u_org,[1 1 1],ind);
        disp(['i = ', num2str(i), ': PSNR = ', num2str(psnr,4), ', SSIM = ', num2str(ssim,4),...
            ', error = ', num2str(error,4)])
    end   
end

%% result plot
ind = u>0;
psnr = EvalImgQuality(u, u_org, 'PSNR');
ssim = ssim_index3d(255*u, 255*u_org, [1 1 1], ind);

visband = [8,16,32];
umax = max(max(max(u_org(:,:,visband))));
plotshape = [1, 3];
ImgPlot(u_org(:,:,visband)/umax, 'groundtruth', 1, [plotshape, 1]);
ImgPlot(v(:,:,visband)/umax, 'observation', 1, [plotshape, 2]);
ImgPlot(u(:,:,visband)/umax, 'result', 1, [plotshape, 3]);

end
