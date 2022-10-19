%%%% Hyperspectral Image Restoration using Hybrid Spatio-Spectral Total Variation via ADMM %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Author: Saori Takeyama (takeyama.s.aa@m.titech.ac.jp)
% Last version: July 30, 2020
% Article: S. Takeyama, S. Ono, and I. Kumazawa, "A Constrained Convex Optimization Approach
% to Hyperspectral Image Restoration with Hybrid Spatio-Spectral Regularization,"
% Remote Sens. vol.12, issue 21, pp. 3541, Oct. 2020.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
addpath subfunctions

%% User Settings %%
mode = 'denoise'; % choose 'denoise' or 'CS'
load 'Beltsville'

sigma = 0.1; % the standard deviation of gaussian noise
s_p = 0.05; % the rate of salt-and-pepper noise (denoise case)
l_v = 0.05; % the rate of vertical line noise (denoise case)
l_h = 0.05; % the rate of horizontal line noise (denoise case)
L = 'L1'; % p = 1 -> 'L1', p = 2 -> 'L1,2'
omega = 0.04; % the parameter in HSSTV
r = 0.2; % randomsampling rate (CS case)

%% main %%
switch mode
    case 'denoise'
        [psnr, ssim] = denoising(sigma, s_p, l_v, l_h, omega, u_org, L);
    case 'CS'
        [psnr, ssim] = CS(sigma, omega, u_org, L, r);
end

%% dispray results %%
disp(['Output PSNR = ', num2str(psnr),' ssim = ', num2str(ssim)]);

            