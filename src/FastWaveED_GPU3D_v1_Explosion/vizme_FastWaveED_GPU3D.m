% Written by Yury Alkhimenkov
% Massachusetts Institute of Technology
% 01 Aug 2023

% This script is developed to run the CUDA C routine "FastWaveED_GPU3D_v1.cu" and visualize the results

% This script:
% 1) creates parameter files
% 2) compiles and runs the code on a GPU
% 3) visualize and save the result
%%

clear
format compact
ScaleT = 1;
DAT = 'double'; 
DAT = 'single';
% NBX is the input parameter to GPU
NBX = ScaleT*12;
NBY = ScaleT*12/3;
NBZ = ScaleT*5;
nt  = 3000 +   0*3550; % number of iterations

BLOCK_X  = 32; % BLOCK_X*BLOCK_Y<=1024
BLOCK_Y  = 2;
BLOCK_Z  = 8;
GRID_X   = NBX*2;
GRID_Y   = NBY*32;
GRID_Z   = NBZ*8;

OVERLENGTH_X = 0;
OVERLENGTH_Y = OVERLENGTH_X;
OVERLENGTH_Z = OVERLENGTH_X;

nx = BLOCK_X*GRID_X  - OVERLENGTH_X; % size of the model in x
ny = BLOCK_Y*GRID_Y  - OVERLENGTH_Y; % size of the model in y
nz = BLOCK_Z*GRID_Z  - OVERLENGTH_Z; % size of the model in z

OVERX = OVERLENGTH_X;
OVERY = OVERLENGTH_Y;
OVERZ = OVERLENGTH_Z;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Physics
% parameters with independent units
spatial_scale = 1;
Lx     = 7680 + 0*9.35/spatial_scale; % m
Ly     = fix(7680/3) + 0*9.35/spatial_scale; % m
Lz     = 3200 + 0*9.35/spatial_scale; % m
K_dry  = 0.2 + 2/3*0.4;      % Bulk m. Pa
G0     = 0.2;                % shear m.
c      = zeros(6,6);
%Set the medium
% 1 --- Isotropic medium
% 2 --- Orthorhombic medium (Glass/Epoxy)
Medium_type = 1;
%% Isotropic rock
if Medium_type == 1
    % elastic constants:
    K   = 24.*1e9;
    G0 = 12.*1e9;
    c(1,1)    = K + 4/3*G0;
    c(2,2)    = K + 4/3*G0;
    c(1,3)    = K -  2/3*G0;
    c(1,2)    = K -  2/3*G0;
    c(2,3)    = K -  2/3*G0;
    c(3,3)    = K + 4/3*G0;
    c(4,4)    = G0;
    c(5,5)    = G0;
    c(6,6)    = G0;
    
    Tor1      = 2.0; % tortuosity
    Tor2      = 2.0; % tortuosity
    Tor3      = 2.0; % tortuosity
    
    rho_solid = 1815;  % solid density
    fi        = 0.2;   % porosity
    rho_fluid = 1040 ; % fluid density
    K_g       = 40e9;  % solid bulk m.
    K_fl      = 2.5e9; % fluid bulk m.
    k_etaf1   = 600*10^(-15) /(1e-3); % permeability/viscosity
    k_etaf2   = 600*10^(-15) /(1e-3); % permeability/viscosity
    k_etaf3   = 600*10^(-15) /(1e-3); % permeability/viscosity
    invisc    = 1; % medium is not inviscid
end
%% Glass/Epoxy
if Medium_type == 2
    % elastic constants:
    c(1,1)    = 39.4e9;
    c(1,2)    = 1.0e9;
    c(1,3)    = 5.8e9;
    c(2,2)    = 39.4e9;
    c(2,3)    = 5.8e9;
    c(3,3)    = 13.1e9;
    c(4,4)    = 3.0e9;
    c(5,5)    = 3.0e9;
    c(6,6)    = 2.0e9;
    
    Tor1      = 2.0; % tortuosity
    Tor2      = 2.0; % tortuosity
    Tor3      = 3.6; % tortuosity
    
    rho_solid = 1815;  % solid density
    fi        = 0.2;   % porosity
    rho_fluid = 1040;  % fluid density
    K_g       = 40e9;  % solid bulk m.
    K_fl      = 2.5e9; % fluid bulk m.
    k_etaf1   = 600*10^(-15) /(1e-3); % permeability/viscosity
    k_etaf2   = 600*10^(-15) /(1e-3); % permeability/viscosity
    k_etaf3   = 100*10^(-15) /(1e-3); % permeability/viscosity
    invisc    = 1; % medium is not inviscid
end
%% Spatial size of the initial condition
lamx    = Lx/spatial_scale/10;
lamy    = Ly/spatial_scale/10;
lamz    = Lz/spatial_scale/10;
%% Calculation of more complex parameters
beta_g  = 1/K_g;
beta_f  = 1/K_fl;
alpha1  = 1 - (c(1,1) + c(1,2) + c(1,3)).*beta_g/3; % Biot coef
alpha2  = 1 - (c(1,3) + c(2,2) + c(2,3)).*beta_g/3; % Biot coef
alpha3  = 1 - (c(1,3) + c(1,3) + c(3,3)).*beta_g/3; % Biot coef
rho     = (1-fi)*rho_solid + fi*rho_fluid;
M1      = (K_g.^2) ./ (    K_g.*(1 + fi.*(K_g/K_fl - 1) ) - ( 2.*c(1,1) + c(3,3) + 2*c(1,2) + 4*c(1,3) )/9    );
%%
beta_d  = 1/K_dry;
alphaIS = 1 - beta_g./beta_d;
GPU_x   = (beta_d - beta_g) / (beta_d - beta_g + fi*(beta_f - beta_g));
K_u     = K_dry/(1 - GPU_x*alphaIS );
M_is    = GPU_x*K_u/alphaIS; %M1     = M_is; to double-check
%%
M11     = (K_g.^2) ./ (    K_g.*(1 + fi.*(K_g./K_fl - 1) ) - ( c(1,1)+c(2,2) + c(3,3)+ 2.*(c(1,2) + c(1,3)  +c(2,3)) )./9    );
%%
c11u    = c(1,1) + 0*alpha1.^2*M1;
c22u    = c(2,2) + 0*alpha2.^2*M1;
c33u    = c(3,3) + 0*alpha3.^2*M1;
c13u    = c(1,3) + 0*alpha1*alpha3*M1;
c12u    = c(1,2) + 0*alpha1*alpha2*M1;
c23u    = c(2,3) + 0*alpha2*alpha3*M1;
c44u    = c(4,4);
c55u    = c(5,5);
c66u    = c(6,6);

mm1     = rho_fluid*Tor1/fi;
mm2     = rho_fluid*Tor2/fi;
mm3     = rho_fluid*Tor3/fi;
delta1  = rho.*mm1 - rho_fluid.^2;
delta2  = rho.*mm2 - rho_fluid.^2;
delta3  = rho.*mm3 - rho_fluid.^2;

eta_k1  = 1./k_etaf1;
eta_k2  = 1./k_etaf2;
eta_k3  = 1./k_etaf3;

dx      = Lx/(nx-1);
dy      = Ly/(ny-1);
dz      = Lz/(nz-1);
%%
iM_ELan = [c11u alpha1*M1;alpha1*M1 M1];
iMdvp   = [mm1 rho_fluid; rho_fluid rho]./delta1;
A11     = iM_ELan(1,1);    A12 = iM_ELan(1,2);   A22 = iM_ELan(2,2); % elast
R11     = iMdvp(1,1); R12 = iMdvp(1,2);R22 = iMdvp(2,2); %densit
A3_m    = -(R11 * R22 - R12 ^ 2) * (A11 * A22 - A12 ^ 2);
A2_m    =  (A11 * R11 - 2 * A12 * R12 + A22 * R22) ;
Solution1inf  = 1./ (( -A2_m + (A2_m.*A2_m + 4.*A3_m)^0.5 )./2./A3_m ).^0.5;
Vp_HF   = Solution1inf;
%%
Vp      = sqrt(c11u/rho);
dt_sound = 1/sqrt(1./dx.^2 + 1./dy.^2 + 1./dz.^2) * 1/Vp_HF ;
dt      = dt_sound*0.6;

%Vp       = sqrt(  (max(K1,K2) + 4/3*max(G1,G2))./min(rho1,rho2)   ); % P-wave velocity
dt       = 1./sqrt(1./dx.^2 + 1./dy.^2 + 1./dz.^2) .* 1/Vp ;
%% Coolecting input parameters for C-Cuda
delta1_av     = 1./delta1;
delta2_av     = 1./delta2;
delta3_av     = 1./delta3;
eta_k1_av     = eta_k1;
eta_k2_av     = eta_k2;
eta_k3_av     = eta_k3;
rho_fluid1_av = rho_fluid;
rho_fluid2_av = rho_fluid;
rho_fluid3_av = rho_fluid;
rho1_av       = rho;
rho2_av       = rho;
rho3_av       = rho;

%vrho_x_11 = mm1./delta1;
%vrho_y_11 = mm2./delta2;
%vrho_z_11 = mm3./delta3;

%%
%%
%% Material properties %c11      = K + 4/3*G; %c13      = K - 2/3*G;
%background=layer 1
rho1      = 2500;          % density
K1        = 1.6*15*1e9;    % Bulk  modulus [GPa]
G1        = 12*1e9;        % Shear modulus [GPa]
c11       = (K1 + 4/3*G1) ;
c13       = (K1 - 2/3*G1)  ;
c66       = G1            ;
Vp1 = sqrt( (K1 + 4/3*G1)./rho1 );
Vs1 = sqrt( ( G1)./rho1 );

% layer 2
rho2      = 2500;      % density
K2        = 0.6*15*1e9;    % Bulk  modulus [GPa]
G2        = 12*1e9;    % Shear modulus [GPa]
rho_n     = rho1         ;
c11_L2       = (K2 + 4/3*G2) ;
c13_L2       = (K2 - 2/3*G2)  ;

Vp2 = sqrt( (K2 + 4/3*G2)./rho2 );
Vs2 = sqrt( ( G2)./rho1 );

dt_check = dx/sqrt(3) / Vp1;

rho =rho1;
dt  = 0.001 / ScaleT *1;

c11u=c11; c33u=c11; c13u=c13; c12u=c13; c23u=c13; c44u=G1; c55u=G1; c66u=G1;
%%
vrho_x_11 = 1./rho;
vrho_y_11 = 1./rho;
vrho_z_11 = 1./rho;

vrho_x_12 = rho_fluid./delta1;
vrho_y_12 = rho_fluid./delta2;
vrho_z_12 = rho_fluid./delta3;

vrho_x_22 = rho./delta1;
vrho_y_22 = rho./delta2;
vrho_z_22 = rho./delta3;
%%
[x, y, z]      = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2,-Lz/2:dz:Lz/2);
vrho_x_11_h = vrho_x_11 + vrho_x_11.*0.5.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2);
vrho_x_11_h = (vrho_x_11_h(1:end-1,:,:)+vrho_x_11_h(1:end-1,:,:))./2;
fid           = fopen('vrho_x_11_hc.dat','wb'); fwrite(fid,vrho_x_11_h(:),DAT); fclose(fid);

c11u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c11u;
c11u_het(:,:,1:200)  = c11u_het(:,:,1:200) *0  + c11_L2;
fid           = fopen('c11u_het.dat','wb'); fwrite(fid,c11u_het(:),DAT); fclose(fid);

c12u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c12u;
c12u_het(:,:,1:200)  = c12u_het(:,:,1:200)*0 + c13_L2 ;
fid           = fopen('c12u_het.dat','wb'); fwrite(fid,c12u_het(:),DAT); fclose(fid);
%%
%% Attenuation
% abs_thick = min(floor(0.2*nx), floor(0.2*nx));         % thicknes of the layer
% abs_thick = 118;%128
% abs_rate = 0.22/abs_thick;      % decay rate
% 
% lmargin = [abs_thick abs_thick abs_thick];
% rmargin = lmargin;
% marginY = floor(0.42*ny);
% marginY = 118;
% weights = ones(nx,ny,nz);
% 
% for iy = 1:ny
%     for ix = 1:nx
%         for iz = 1:nz
%         i = 0;
%         j = 0;
%         k = 0;
%         if (ix < lmargin(1) + 1)
%             i = lmargin(1) + 1 - ix;
%         end
%         if (iy < marginY + 1)
%             k = marginY + 1 - iy;
%         end
%         %if (iz < lmargin(3) + 1)
%         %    j = lmargin(3) + 1 - iz;
%         %end
%         if (nx - rmargin(1) < ix)
%             i = ix - nx + rmargin(1);
%         end
%         if (ny - marginY < iy)
%             k = iy - ny + marginY;
%         end
%         if (nz - rmargin(3) < iz)
%             j = iz - nz + rmargin(3);
%         end
%         if (i == 0 && j == 0 && k == 0)
%             continue
%         end
%         rr = abs_rate * abs_rate * double(i*i + j*j + k*k );
%         weights(ix,iy,iz) = exp(-rr);
%         end
%     end
% end
%plot(weights(:,150))
%weights(1:250,:,:) = 
%%
lpml = 110;
pml_width = lpml;
R = 4e-4;
%dt = 0.001; % Assuming dt is defined somewhere in your code

% Precompute sigma and pml arrays using vectorized operations
i_vec = (1:lpml).';
sigma = 3 * 1/2 / (pml_width / nx) * log10(1 / R) .* ((pml_width - i_vec + 1) / pml_width) .^ 2;

pml_x = zeros(nx, 1);
pml_y = zeros(ny, 1);
pml_z = zeros(nz, 1);

pml_x(1:lpml) = sigma;
pml_x(end-lpml+1:end) = flip(sigma);
pml_y(1:lpml) = sigma;
pml_y(end-lpml+1:end) = flip(sigma);
pml_z(1:lpml) = sigma;
pml_z(end-lpml+1:end) = flip(sigma);

% Precompute damping factors
damping_factor_x = exp(-pml_x(1:lpml) * dt);
damping_factor_y = exp(-pml_y(1:lpml) * dt);
damping_factor_z = exp(-pml_z(1:lpml) * dt);

% Initialize weights
weights_x = ones(nx+1, ny, nz);
weights_y = ones(nx, ny+1, nz);
weights_z = ones(nx, ny, nz+1);
weights_xy = ones(nx+1, ny+1, nz);
weights_xz = ones(nx+1, ny, nz+1);
weights_yz = ones(nx, ny+1, nz+1);

% Assign weights using vectorized operations
weights_x(1:lpml, :, :) = repmat(damping_factor_x, [1, ny, nz]);
weights_x(end-lpml+1:end, :, :) = repmat(flip(damping_factor_x), [1, ny, nz]);
weights_y(:, 1:lpml, :) = repmat(damping_factor_y.', [nx, 1, nz]);
weights_y(:, end-lpml+1:end, :) = repmat(flip(damping_factor_y).', [nx, 1, nz]);
weights_z(:, :, end-lpml+1:end) = repmat(reshape(flip(damping_factor_z), [1, 1, lpml]), [nx, ny, 1]);

for i = 1:lpml
    for j = 1:lpml
        damping_factor_xy = exp(-0.5 * (pml_x(i) + pml_y(j)) * dt);
        damping_factor_xz = exp(-0.5 * (pml_x(i) + pml_z(j)) * dt);
        damping_factor_yz = exp(-0.5 * (pml_y(i) + pml_z(j)) * dt);

        weights_xy(i, pml_width:end-pml_width, :) = damping_factor_xy;
        weights_xy(pml_width:end-pml_width, j, :) = damping_factor_xy;
        weights_xy(i, j, :) = damping_factor_xy;
        weights_xy(nx-i, pml_width:end-pml_width, :) = damping_factor_xy;
        weights_xy(pml_width:end-pml_width, ny-j, :) = damping_factor_xy;
        weights_xy(nx-i, ny-j, :) = damping_factor_xy;
        weights_xy(nx-i, j:pml_width, :) = damping_factor_xy;
        weights_xy(i:pml_width, ny-j, :) = damping_factor_xy;

        weights_xz(i, :, 1:end-pml_width) = damping_factor_xz;
        weights_xz(nx-i, :, 1:end-pml_width) = damping_factor_xz;
        weights_xz(pml_width:end-pml_width, :, nz-j) = damping_factor_xz;
        weights_xz(nx-i, :, nz-j) = damping_factor_xz;
        weights_xz(i:pml_width, :, nz-j) = damping_factor_xz;

        weights_yz(:, i, 1:end-pml_width) = damping_factor_yz;
        weights_yz(:, ny-i, 1:end-pml_width) = damping_factor_yz;
        weights_yz(:, pml_width:end-pml_width, nz-j) = damping_factor_yz;
        weights_yz(:, ny-i, nz-j) = damping_factor_yz;
        weights_yz(:, i:pml_width, nz-j) = damping_factor_yz;
    end
end



figure(3);clf ;colormap(jet);%colormap(Red_blue_colormap);
plot(1:pml_width,sigma(1:pml_width),'-r')
drawnow;
%%
figure(1);colormap(jet);clf
Pl = squeeze (weights_xy(:,:,150));
mesh(Pl);view(0,90),colorbar;drawnow;
fid           = fopen('weights_x.dat','wb'); fwrite(fid,weights_x(:),DAT); fclose(fid);
fid           = fopen('weights_y.dat','wb'); fwrite(fid,weights_y(:),DAT); fclose(fid);
fid           = fopen('weights_z.dat','wb'); fwrite(fid,weights_z(:),DAT); fclose(fid);
fid           = fopen('weights_xy.dat','wb'); fwrite(fid,weights_xy(:),DAT); fclose(fid);
fid           = fopen('weights_xz.dat','wb'); fwrite(fid,weights_xz(:),DAT); fclose(fid);
fid           = fopen('weights_yz.dat','wb'); fwrite(fid,weights_yz(:),DAT); fclose(fid);
%%
pa1           = [dx dy dz dt Lx Ly Lz lamx lamy lamz];
pa2           = [c11u c33u c13u c12u c23u c44u c55u c66u alpha1 alpha2 alpha3 M1 c22u];
%pa3           = [ vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22];
pa3           = [vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22 eta_k1_av eta_k2_av eta_k3_av ];
fid           = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid           = fopen('pa2.dat','wb'); fwrite(fid,pa2(:),DAT); fclose(fid);
fid           = fopen('pa3.dat','wb'); fwrite(fid,pa3(:),DAT); fclose(fid);
%%
% srctimefun = load('srctimefun.txt');
% % figure(6);clf; plot(srctimefun);
% nt = 1050;
% Src   = zeros(nt  ,1);
% Src(1:101) = srctimefun;
% figure(6);clf; plot(Src);

srctimefun = load('srctimefun.txt');
srctimefun = 3.3*10e10*interp1(1:101,srctimefun,1:0.5:100);%3.3e20*
S = size(srctimefun);S = S(2);
SourceSharp = zeros(1,nt); SourceSharp(120:S(1)+119) = srctimefun;

[x, y]   = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2);
lamx = 350;
init = 7*2e9*exp(-(x(1:end,1)/lamx).^2 );
SourceGauss = zeros(1,nt); SourceGauss(1:nx-99) = init(100:end);
 %SourceSharp(1:140) = max(SourceSharp(1:140) ,SourceGauss(1:140) );
% SourceSharp(300:end) = max(SourceSharp(300:end)  ,SourceGauss(300:end)  );
lamx = lamx;
init = 3.3*2e9*exp(-(x(1:end,1)/lamx).^2 );
SourceGauss = zeros(1,nt); SourceGauss(1:nx-101) = init(102:end);
%SourceSharp(180:260) = max(SourceSharp(180:260) ,SourceGauss(180:260));
GGG = SourceGauss(1:nt);
%SourceSharp(:) = max(SourceSharp(:) ,GGG(:));
%SourceSharp(350:end) = min(SourceSharp(350:end) ,GGG(350:end));

% p = polyfit(1:nt,SourceSharp,9);
% x1 = [1:nt];
% SourceSharp = polyval(p,x1); SourceSharp(1:90)=0;
% SourceSharp(359:end)=0;
% SourceSharp(1:end) = max(SourceSharp(1:end) ,SourceGauss(1:end));

 %SourceSharp = smooth(SourceSharp) ;%SourceSharp = smooth(SourceSharp) ;
 %SourceSharp = smooth(SourceSharp) ;SourceSharp = smooth(SourceSharp) ;
% SourceSharp = smooth(SourceSharp) ;SourceSharp = smooth(SourceSharp) ;
% SourceSharp = smooth(SourceSharp) ;SourceSharp = smooth(SourceSharp) ;

SourceGauss = zeros(1,nt); SourceGauss(1:nx-199) = init(200:end);
 SourceSharp(1:140) = max(SourceSharp(1:140) ,SourceGauss(1:140) );
%  SourceSharp(300:end) = max(SourceSharp(300:end)  ,SourceGauss(300:end)  );
% SourceSharp(180:260) = max(SourceSharp(180:260) ,SourceGauss(180:260) );
% for ii=1:4
%   SourceSharp = smoothdata(SourceSharp,'gaussian',15) ;
%   SourceSharp = smoothdata(SourceSharp,'gaussian',15) ;
% end

t        = (0:nt-1)*dt;

max_SourceGauss = max(SourceGauss(:));
%SourceGauss = 1e9.*SourceGauss./max_SourceGauss;
%SourceSharp = 1e9.*SourceSharp./max_SourceGauss;

%Src = SourceSharp;
Src = SourceGauss';
figure(6);clf;plot(Src,'r');hold on;plot(SourceGauss,'b');
figure(2);clf;plot(diff(SourceSharp,2),'r');hold on;plot(diff(SourceGauss,2),'b');
% 
zz=1;
% Fsh=abs(fft(SourceSharp)); Fg = abs(fft(SourceGauss));
% figure(1);clf;plot(Fsh,'r');hold on;plot(Fg,'b');
% Fsh(33:500)=Fg(33:500); Fsh(1000:2517)=Fg(1000:2517); figure(1);clf;plot(Fsh,'r');hold on;plot(Fg,'b');
% FshINV = real(ifft(Fsh)); FshINV_s=FshINV*0; 
% FshINV_s(222:721)=FshINV(1:500); 
% FshINV_s(1:222)=FshINV(end-221:end);
% FshINV_s(1:140) = max(FshINV_s(1:140) ,SourceGauss(1:140)' ); FshINV_s(1:100) = SourceGauss(1:100)' ;
% FshINV_s(300:end) = max(FshINV_s(300:end)  ,SourceGauss(300:end)'  );FshINV_s(350:end) = SourceGauss(350:end)';
% figure(1);clf;plot(FshINV_s','r');hold on;plot(SourceGauss,'b');
% %Src = FshINV_s;

fid           = fopen('Src.dat','wb'); fwrite(fid,Src(:),DAT); fclose(fid);
%% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
code_name    = 'FastWaveED_GPU3D_v1';
run_cmd      =['nvcc -arch=sm_80 -O3' ...
    ' -DNBX='    int2str(NBX)         ...
    ' -DNBY='    int2str(NBY)         ...
    ' -DNBZ='    int2str(NBZ)         ...
    ' -DOVERX='  int2str(OVERX)       ...
    ' -DOVERY='  int2str(OVERY)       ...
    ' -DOVERZ='  int2str(OVERZ)       ...
    ' -DNPARS1=' int2str(length(pa1)) ...
    ' -DNPARS2=' int2str(length(pa2)) ...
    ' -DNPARS3=' int2str(length(pa3)) ' ',code_name,'.cu'];
%    ' -Dnt='     int2str(nt)          ...

delete a.exe
%! module load cuda/10.0
system(run_cmd);
tic;

! a.exe
GPU_time = toc

%% to run from the terminal
% nvcc -arch=sm_52 -O3 -DNBX=4 -DNBY=4 -DNBZ=4 -DOVERX=0 -DOVERY=0 -DOVERZ=0 -Dnt=300 -DNPARS1=10 -DNPARS2=13 -DNPARS3=15  GPU3D_Biot_v2.cu
%% Reading data
% Load the DATA and infos
isave = 0;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end
name=[num2str(isave) '_0_Vx.res']; id = fopen(name); Vx  = fread(id,DAT); fclose(id); Vx  = reshape(Vx  ,nx+1,ny  ,nz  );
%name=[num2str(isave) '_0_Vy.res']; id = fopen(name); Vy  = fread(id,DAT); fclose(id); Vy  = reshape(Vy  ,nx  ,ny+1,nz  );
name=[num2str(isave) '_0_Vz.res']; id = fopen(name); Vz  = fread(id,DAT); fclose(id); Vz  = reshape(Vz  ,nx  ,ny  ,nz+1);
name=[num2str(isave) '_0_Src.res']; id = fopen(name); Src   = fread(id,DAT); fclose(id); Src   = reshape(Src  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec.res']; id = fopen(name); Rec   = fread(id,DAT); fclose(id); Rec  = reshape(Rec  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_x.res']; id = fopen(name); Rec_x   = fread(id,DAT); fclose(id); Rec_x  = reshape(Rec_x  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y.res']; id = fopen(name); Rec_y   = fread(id,DAT); fclose(id); Rec_y  = reshape(Rec_y  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y2.res']; id = fopen(name); Rec_y2   = fread(id,DAT); fclose(id); Rec_y2  = reshape(Rec_y2  ,nt  ,1  ,1);

%DAT = 'double';
% name=[num2str(isave) '_0_Rec1.res']; id = fopen(name); Rec1   = fread(id,DAT); fclose(id); Rec1  = reshape(Rec1  ,nt*0+1600*2  ,1  ,1);
% name=[num2str(isave) '_0_Rec_x1.res']; id = fopen(name); Rec_x1   = fread(id,DAT); fclose(id); Rec_x1  = reshape(Rec_x1  ,nt*0+1600*2  ,1  ,1);
% name=[num2str(isave) '_0_Rec_y1.res']; id = fopen(name); Rec_y1   = fread(id,DAT); fclose(id); Rec_y1  = reshape(Rec_y1  ,nt*0+1600*2  ,1  ,1);

%name=[num2str(isave) '_0_sigma_xx.res']; id = fopen(name); sigma_xx  = fread(id,DAT); fclose(id); sigma_xx  = reshape(sigma_xx  ,nx,ny  ,nz  );
% name=[num2str(isave) '_0_xx.res'];  id = fopen(name); XX   = fread(id,DAT); fclose(id); XX   = reshape(XX  ,nx  ,ny  ,nz  );
% name=[num2str(isave) '_0_yy.res'];  id = fopen(name); YY   = fread(id,DAT); fclose(id); YY   = reshape(YY  ,nx  ,ny  ,nz  );
% name=[num2str(isave) '_0_zz.res'];  id = fopen(name); ZZ   = fread(id,DAT); fclose(id); ZZ   = reshape(ZZ  ,nx  ,ny  ,nz  );
%%
figure(2),clf,
plot(Src); title('Source GPU');
%% m0=3.3e20; mt=[0,0,1; 0,0,0.8; 1,0.8,0]; dura=0.2; az=0.0; outnm ='synseis'; nam='vsimple_1.0/2.70.grn.';
figure(4),clf,
Max_rec = 2048*1;
isyn_c_nameZ = sprintf('synseisExpl.z.txt');isyn_cZ = load(isyn_c_nameZ);%DCsL Expl Coll
delay = 1440;synseis_delay = zeros(1,nt);synseis_delay(delay+1:Max_rec+delay)=isyn_cZ(1: [Max_rec]) ;

isyn_c_nameX = sprintf('synseisExpl.r.txt');isyn_cX = load(isyn_c_nameX);
synseis_delayX = zeros(1,nt);synseis_delayX(delay+1:Max_rec+delay)=isyn_cX(1: [Max_rec]) ;

isyn_c_nameY = sprintf('synseisExpl.t.txt');isyn_cY = load(isyn_c_nameY);
synseis_delayY = zeros(1,nt);synseis_delayY(delay+1:Max_rec+delay)=isyn_cY(1: [Max_rec]) ;

Ampl = 3. *0.81*2e4 *4.9 *0.5   *0.9*0.1;
Rec2=Ampl*Rec; RecX = Ampl*Rec_x;
RecY = Ampl*( Rec_y*2 + 0*Rec_y2)*0.5; 
%RecY = Ampl*( Rec_y + 0*Rec_y2)*1; 

% fid  = fopen('Rec2.txt','w'); fprintf(fid,'%14.12f\n',Rec2);  fclose(fid);
% id= fopen('Rec2.txt'); data = textscan(id,'%s'); fclose(id); Rec2 = str2double(data{1}(1:1:end));
% 
% fid  = fopen('synseis_delay.txt','w'); fprintf(fid,'%14.12f\n',synseis_delay);  fclose(fid);
% id= fopen('synseis_delay.txt'); data = textscan(id,'%s'); fclose(id); synseis_delay = str2double(data{1}(1:1:end));

for iii=1:15
%synseis_delay = smoothdata(synseis_delay,'gaussian',25) ; 
end
%plot(synseis_delay,'bo-'); hold on; plot(Rec2,'r-','LineWidth',2.5); hold on;  title('Reciever GPU');
%plot(Rec,'r-','LineWidth',2.5); hold on;  plot(Rec_x,'b-.','LineWidth',2.5); hold on;  plot(Rec_y,'g-.','LineWidth',2.5); 
end_minus = 0;
size_x = size(synseis_delay(1000:end-500));
x_timeZHU =( 1001:1:(size_x(2) +1000) )'.*dt;
x_timeGPU =( 1000:1:(nt-end_minus) )'.*dt;


figure(4),clf
end_minus = 200;
start_Zhu = 1000*0 +511; end_Zhu = 1;
size_x = size(synseis_delay(start_Zhu:end-end_Zhu));

start_num = 1000*0 + 1; start_num2 = 1;
x_timeGPU =( start_num:1:(nt-end_minus) )'.*dt;
x_timeZHU =( start_num:1:(size_x(2) - 0*end_minus) )'.*dt;

Rec2(2350:end)=0;RecX(2350:end)=0;RecY(2350:end)=0;
subplot(1,3,1); plot(x_timeZHU,1*1e0*synseis_delay(start_Zhu:end-end_Zhu),'r-','LineWidth',3.0); hold on; plot(x_timeGPU(1:end-start_num2),Rec2((start_num+start_num2):end-end_minus),'b-.','LineWidth',1.5); title('Vertical'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWaveED GPU3D');ylim([-1250 1250]); 
xlim([0.0 2.5]);      %Rec2(1000:end-end_minus)
%hold on; plot(Ampl.*Rec1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

subplot(1,3,2); plot(x_timeZHU,(1*1e0*synseis_delayX(start_Zhu:end-end_Zhu)),'r-','LineWidth',3.0); hold on; plot(x_timeGPU(1:end-start_num2),(-RecX((start_num+start_num2):end-end_minus)),'b-.','LineWidth',1.5); title('Radial'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWaveED GPU3D'); ylim([-1250 1250]); 
xlim([0.0 2.5]);    
%hold on; plot((-Ampl.*Rec_x1(1000:end-end_minus)),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

subplot(1,3,3); plot(x_timeZHU,1*1e0*synseis_delayY(start_Zhu:end-end_Zhu),'r-','LineWidth',3.0); hold on; plot(x_timeGPU(1:end-start_num2),-RecY((start_num+start_num2):end-end_minus),'b-.','LineWidth',1.5); title('Transverse'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWaveED GPU3D'); ylim([-1250 1250]);  
xlim([0.0 2.5]);   
%hold on; plot(-Ampl.*Rec_y1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); %ylabel('  1/Q','fontsize',18); 
legend boxoff

zzz = 1;
if 4>3; return; end
delete *.res *.inf  a.exe *.dat
