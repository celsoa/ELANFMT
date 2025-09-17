%%
% Written by Yury Alkhimenkov
% Massachusetts Institute of Technology
% 01 Aug 2023

% This script is developed to run the CUDA C routine "FastWaveED_GPU3D_v1.cu" and visualize the results

% This script:
% 1) creates parameter files
% 2) compiles and runs the code on a GPU
% 3) visualize and save the result
clear
format compact
ScaleT = 1;
% NBX is the input parameter to GPU
NBX = ScaleT*12;
NBY = ScaleT*12;
NBZ = ScaleT*7;
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
Ly     = 7680 + 0*9.35/spatial_scale; % m
Lz     = 4480 + 0*9.35/spatial_scale; % m
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

dt_check = 1/sqrt(3) / Vp1;

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
fid           = fopen('vrho_x_11_hc.dat','wb'); fwrite(fid,vrho_x_11_h(:),'single'); fclose(fid);

c11u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c11u;
c11u_het(:,:,1:200)  = c11u_het(:,:,1:200) *0  + c11_L2;
c11u_het(:,:,1) = 0;
fid           = fopen('c11u_het.dat','wb'); fwrite(fid,c11u_het(:),'single'); fclose(fid);

c12u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c12u;
c12u_het(:,:,1:200)  = c12u_het(:,:,1:200)*0 + c13_L2 ;
c12u_het(:,:,1) = 0;
fid           = fopen('c12u_het.dat','wb'); fwrite(fid,c12u_het(:),'single'); fclose(fid);

%c44u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c44u;
%fid           = fopen('c44u_het.dat','wb'); fwrite(fid,c44u_het(:),'single'); fclose(fid);
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
% %plot(weights(:,150))
% %weights(1:250,:,:) = 
% figure(1)
% Pl = squeeze (weights(:,150,:));
% mesh(Pl);view(0,90),colorbar
%fid           = fopen('weights.dat','wb'); fwrite(fid,weights(:),'single'); fclose(fid);
%% NEW PML
weights_x = ones(nx+1,1);
weights_y = ones(ny+1,1);
weights_z = ones(nz+1,1);
pml_x = zeros(nx, 1);
pml_y = zeros(ny, 1);
pml_z = zeros(nz, 1);
lpml = 110; 
pml_width = lpml;
for i = 1:lpml
    R = 1e-8;
    %sigma = sigma_max * ((pml_width - i + 1) / pml_width)^2;
    sigma_x(i) =  3*1/2/(pml_width/nx).*log10(1/R) .* ((pml_width - i + 1) / pml_width)^2;
    sigma_y(i) =  3*1/2/(pml_width/ny).*log10(1/R) .* ((pml_width - i + 1) / pml_width)^2;
    sigma_z(i) =  3*1/2/(pml_width/nz).*log10(1/R) .* ((pml_width - i + 1) / pml_width)^2;
    pml_x(i) = sigma_x(i);
    pml_x(nx-i+1) = sigma_x(i);
    pml_y(i) = sigma_y(i);
    pml_y(ny-i+1) = sigma_y(i);

    pml_z(i) = sigma_z(i);
    pml_z(nz-i+1) = sigma_z(i);
end
for i = 1:lpml
    for j = 1:lpml
        damping_factor_x = exp(-pml_x(i) * dt);
        weights_x(i) = damping_factor_x;
        damping_factor_y =  exp(-pml_y(j) * dt);
        weights_y(j) = damping_factor_y;

        damping_factor_x = exp(-pml_x(i) * dt);
        weights_x(nx-i+2) = damping_factor_x;
        damping_factor_y =  exp(-pml_y(j) * dt);
        weights_y(ny-j+2) = damping_factor_y;

        damping_factor_z =  exp(-pml_z(i) * dt);
        weights_z(i) = damping_factor_z;
        weights_z(nz-i+2) = damping_factor_z;

    end
end
weights_z(1:fix(nz/2))=1;
fid           = fopen('weights_x.dat','wb'); fwrite(fid,weights_x(:),'single'); fclose(fid);
fid           = fopen('weights_y.dat','wb'); fwrite(fid,weights_y(:),'single'); fclose(fid);
fid           = fopen('weights_z.dat','wb'); fwrite(fid,weights_z(:),'single'); fclose(fid);

figure(3);clf ;colormap(jet);%colormap(Red_blue_colormap);
plot(1:pml_width,sigma_x(1:pml_width),'-r' );
%plot( weights_z,'-b' );
drawnow;
%%
pa1           = [dx dy dz dt Lx Ly Lz lamx lamy lamz];
pa2           = [c11u c33u c13u c12u c23u c44u c55u c66u alpha1 alpha2 alpha3 M1 c22u];
%pa3           = [ vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22];
pa3           = [vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22 eta_k1_av eta_k2_av eta_k3_av ];
fid           = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),'single'); fclose(fid);
fid           = fopen('pa2.dat','wb'); fwrite(fid,pa2(:),'single'); fclose(fid);
fid           = fopen('pa3.dat','wb'); fwrite(fid,pa3(:),'single'); fclose(fid);

MTSxx=0; MTSyy=0; MTSzz=0; MTSxy=0; MTSxz=1; MTSyz=0.8; 
MTS           = [MTSxx MTSyy MTSzz MTSxy MTSxz MTSyz ];
fid           = fopen('MTS.dat','wb'); fwrite(fid,MTS(:),'single'); fclose(fid);
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

fid           = fopen('Src.dat','wb'); fwrite(fid,Src(:),'single'); fclose(fid);
%% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
code_name    = 'FastWaveED_GPU3D_v1';
run_cmd      =['nvcc -arch=sm_89 -O3' ...
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

%delete a.exe
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
% name=[num2str(isave) '_0_Vx.res']; id = fopen(name); Vx  = fread(id,DAT); fclose(id); Vx  = reshape(Vx  ,nx+1,ny  ,nz  );
% %name=[num2str(isave) '_0_Vy.res']; id = fopen(name); Vy  = fread(id,DAT); fclose(id); Vy  = reshape(Vy  ,nx  ,ny+1,nz  );
% name=[num2str(isave) '_0_Vz.res']; id = fopen(name); Vz  = fread(id,DAT); fclose(id); Vz  = reshape(Vz  ,nx  ,ny  ,nz+1);
name=[num2str(isave) '_0_Src.res']; id = fopen(name); Src   = fread(id,DAT); fclose(id); Src   = reshape(Src  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec.res']; id = fopen(name); Rec   = fread(id,DAT); fclose(id); Rec  = reshape(Rec  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_x.res']; id = fopen(name); Rec_x   = fread(id,DAT); fclose(id); Rec_x  = reshape(Rec_x  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y.res']; id = fopen(name); Rec_y   = fread(id,DAT); fclose(id); Rec_y  = reshape(Rec_y  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y2.res']; id = fopen(name); Rec_y2   = fread(id,DAT); fclose(id); Rec_y2  = reshape(Rec_y2  ,nt  ,1  ,1);

%name=[num2str(isave) '_0_Rec1.res']; id = fopen(name); Rec1   = fread(id,DAT); fclose(id); Rec1  = reshape(Rec1  ,nt*0+2*1600  ,1  ,1);
%name=[num2str(isave) '_0_Rec_x1.res']; id = fopen(name); Rec_x1   = fread(id,DAT); fclose(id); Rec_x1  = reshape(Rec_x1  ,nt*0+2*1600  ,1  ,1);
%name=[num2str(isave) '_0_Rec_y1.res']; id = fopen(name); Rec_y1   = fread(id,DAT); fclose(id); Rec_y1  = reshape(Rec_y1  ,nt*0+2*1600  ,1  ,1);

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
isyn_c_nameZ = sprintf('synseisDCsL.z.txt');isyn_cZ = load(isyn_c_nameZ);
delay = 1440;synseis_delay = zeros(1,nt);synseis_delay(delay+1:Max_rec+delay)=isyn_cZ(1: [Max_rec]) ;

isyn_c_nameX = sprintf('synseisDCsL.r.txt');isyn_cX = load(isyn_c_nameX);
synseis_delayX = zeros(1,nt);synseis_delayX(delay+1:Max_rec+delay)=isyn_cX(1: [Max_rec]) ;

isyn_c_nameY = sprintf('synseisDCsL.t.txt');isyn_cY = load(isyn_c_nameY);
synseis_delayY = zeros(1,nt);synseis_delayY(delay+1:Max_rec+delay)=isyn_cY(1: [Max_rec]) ;

% Max_rec = 2048*1;
% isyn_c_nameZ = sprintf('synseis2024.z.txt');isyn_cZ = load(isyn_c_nameZ);
% delay = 1440;synseis_delay = zeros(1,nt);synseis_delay(delay+1:Max_rec+delay)=isyn_cZ(1: [Max_rec]) ;
% 
% isyn_c_nameX = sprintf('synseis2024.r.txt');isyn_cX = load(isyn_c_nameX);
% synseis_delayX = zeros(1,nt);synseis_delayX(delay+1:Max_rec+delay)=isyn_cX(1: [Max_rec]) ;
% 
% isyn_c_nameY = sprintf('synseis2024.t.txt');isyn_cY = load(isyn_c_nameY);
% synseis_delayY = zeros(1,nt);synseis_delayY(delay+1:Max_rec+delay)=isyn_cY(1: [Max_rec]) ;


Ampl = 2e1 *4.9 *0.5   *0.9*0.1;
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
clf
%%
figure(5),clf
subplot(1,3,1); %clf
ZhuF=synseis_delay(start_Zhu:end-end_Zhu);
GPUF=1.5/3/1e-1*Rec2((start_num+start_num2):end-end_minus)*0.23;%1.002;
 
 flo = 1/25;%0.065;      % Hz
fhi = 1/12.5;%0.11;      % Hz
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
 fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
 flo = flo/(fs_Hz/2);
 fhi = fhi/(fs_Hz/2);
[b, a] = butter(4, [flo fhi],'bandpass'); 
yG = filtfilt(b, a, GPUF);%yG=GPUF;
yZ = filtfilt(b, a, ZhuF );%yZ=ZhuF(1:4499)';

 [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot( Tsignal(1+0000:end)/1,yZ(1:end-0000),'r-','LineWidth',1.5); hold on;
plot( Tsignal(1+0000:end)/1,yG(1:end-0000),'b-.','LineWidth',1.5); title('Vertical'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');% ylim([-5e-3 5e-3]); xlim([1 5.5]);   %Rec2(1000:end-end_minus)
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-2e-1 2e-1]);
 xlim([0 4.0]);

 %clf
subplot(1,3,2);% clf
ZhuF=(synseis_delayX(start_Zhu:end-end_Zhu));
GPUF=-1.2/3/1e-1*(RecX((start_num+start_num2):end-end_minus))*0.3;%1.3;
 
signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz
  flo = flo/(fs_Hz/2);
 fhi = fhi/(fs_Hz/2);
 [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal/1, yZ,'r-','LineWidth',1.5); hold on;
plot(Tsignal(1+000:end)/1, yG(1:end-000),'b-.','LineWidth',1.5); title('Radial'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');%ylim([-2e-3 2e-3]); xlim([1 5.5]);  
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-2e-1 2e-1]);
 xlim([0 4]); 
%clf
subplot(1,3,3); %clf
ZhuF=synseis_delayY(start_Zhu:end-end_Zhu);
GPUF=-12/3/1e-1*RecY((start_num+start_num2):end-end_minus-0*20)*0.028;
 
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
flo = flo/(fs_Hz/2);
fhi = fhi/(fs_Hz/2);
 
  [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal(1+0:end)/1,yZ(1:end-0),'r-','LineWidth',1.5); hold on;
plot(Tsignal(1+000:end)/1,yG(1:end-000),'b-.','LineWidth',1.5); title('Transverse'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); %ylim([-1e-2 1e-2]); xlim([1 5.5]);   
 xlabel(' Time (s)'); ylabel('v (m/s)'); xlim([0 4]);
 ylim([-2e-1 2e-1]);
legend boxoff
%% SURFACE WAVES



%%
figure(6),clf

subplot(1,3,1); %clf
ZhuF=synseis_delay(start_Zhu:end-end_Zhu);
GPUF=1.5/3/1e-1*Rec2((start_num+start_num2):end-end_minus)*0.23;%1.002;
 
 flo = 1/25;%0.065;      % Hz
fhi = 1/12.5;%0.11;      % Hz
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
 fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
 flo = flo/(fs_Hz/2);
 fhi = fhi/(fs_Hz/2);
[b, a] = butter(4, [flo fhi],'bandpass'); 
yG = filtfilt(b, a, GPUF);%yG=GPUF;
yZ = filtfilt(b, a, ZhuF );%yZ=ZhuF(1:4499)';

 [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot( Tsignal(1+0000:end)/1,yZ(1:end-0000),'r-','LineWidth',1.5); hold on;
plot( Tsignal(1+0000:end)/1,yG(1:end-0000),'b-.','LineWidth',1.5); title('Vertical'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');% ylim([-5e-3 5e-3]); xlim([1 5.5]);   %Rec2(1000:end-end_minus)
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-2e-1 2e-1]);
 xlim([1 6.0]);

 %clf
subplot(1,3,2);% clf
ZhuF=(synseis_delayX(start_Zhu:end-end_Zhu));
GPUF=-1.2/3/1e-1*(RecX((start_num+start_num2):end-end_minus))*0.3;%1.3;
 
signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz
  flo = flo/(fs_Hz/2);
 fhi = fhi/(fs_Hz/2);
 [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal/1, yZ,'r-','LineWidth',1.5); hold on;
plot(Tsignal(1+000:end)/1, yG(1:end-000),'b-.','LineWidth',1.5); title('Radial'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');%ylim([-2e-3 2e-3]); xlim([1 5.5]);  
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-0.1 0.1]);
 xlim([1 6]); 
%clf
subplot(1,3,3); %clf
ZhuF=synseis_delayY(start_Zhu:end-end_Zhu);
GPUF=-12/3/1e-1*RecY((start_num+start_num2):end-end_minus-0*20)*0.028;
 
 flo = 1/2.5;%0.065;      % Hz
 fhi = 1/0.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:2799)=GPUF(1:2799);GPUF=signal;
signal = zeros(150000,1);signal(1:2799)=ZhuF(1:2799)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
flo = flo/(fs_Hz/2);
fhi = fhi/(fs_Hz/2);
 
  [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal(1+0:end)/1,yZ(1:end-0),'r-','LineWidth',1.5); hold on;
plot(Tsignal(1+000:end)/1,yG(1:end-000),'b-.','LineWidth',1.5); title('Transverse'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); %ylim([-1e-2 1e-2]); xlim([1 5.5]);   
 xlabel(' Time (s)'); ylabel('v (m/s)'); xlim([1 6]);
 ylim([-6*1e-2 6*1e-2]);
legend boxoff


%%
figure(4),clf
%Rec2(2350:end)=0;RecX(2350:end)=0;RecY(2350:end)=0;

x_timeGPU =( start_num:1:(nt*2.2-end_minus) )'.*dt;
x_timeZHU =( start_num:1:(size_x(2)*2.2 - 0*end_minus) )'.*dt;
synseis_delayN=( start_num:1:(size_x(2)*3 - 0*end_minus) )'.*0;
synseis_delayN(1:3488)= synseis_delay;
Rec2N=( start_num:1:(nt*3-end_minus) )'.*0;
Rec2N(1:3000)= Rec2;

subplot(1,3,1); plot(x_timeZHU,1*1e0*synseis_delayN(start_Zhu:6549+start_Zhu-1),'r-','LineWidth',1.5); hold on; 
plot(x_timeGPU(1:end-start_num2),Rec2N((start_num+start_num2):6400),'b-.','LineWidth',1.5); title('Vertical'); 
%subplot(1,3,1); plot(x_timeZHU,1*1e0*synseis_delayN(start_Zhu:end-end_Zhu),'r-','LineWidth',1.5); hold on; plot(x_timeGPU(1:end-start_num2),Rec2((start_num+start_num2):end-end_minus),'b-.','LineWidth',1.5); title('Vertical'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-0.8 0.8]); xlim([0.0 4]);      %Rec2(1000:end-end_minus)
%hold on; plot(Ampl.*Rec1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

synseis_delayXN=( start_num:1:(size_x(2)*3 - 0*end_minus) )'.*0;
synseis_delayXN(1:3488)= synseis_delayX;
RecXN=( start_num:1:(nt*3-end_minus) )'.*0;
RecXN(1:3000)= RecX;
subplot(1,3,2); plot(x_timeZHU,(1*1e0*synseis_delayXN(start_Zhu:6549+start_Zhu-1)),'r-','LineWidth',1.5); hold on; 
plot(x_timeGPU(1:end-start_num2),(-RecXN((start_num+start_num2):6400)),'b-.','LineWidth',1.5); title('Radial'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-0.8 0.8]); xlim([0.0 4]);    
%hold on; plot((-Ampl.*Rec_x1(1000:end-end_minus)),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

synseis_delayYN=( start_num:1:(size_x(2)*3 - 0*end_minus) )'.*0;
synseis_delayYN(1:3488)= synseis_delayY;
RecYN=( start_num:1:(nt*3-end_minus) )'.*0;
RecYN(1:3000)= RecY;
subplot(1,3,3); plot(x_timeZHU,1*1e0*synseis_delayYN(start_Zhu:6549+start_Zhu-1),'r-','LineWidth',1.5); hold on; 
plot(x_timeGPU(1:end-start_num2),-RecYN((start_num+start_num2):6400),'b-.','LineWidth',1.5); title('Transverse'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-0.8 0.8]); xlim([0.0 4]);   
%hold on; plot(-Ampl.*Rec_y1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); %ylabel('  1/Q','fontsize',18); 
legend boxoff

zzz = 1;
if 3>2; return; end
delete *.res *.inf  a.exe *.dat a.lib a.exp

%%/scratch/yalkhime/Octopus/2019_Codes/3D_progress/Biot_3D_v7_v7_het/FastBiot_v1.0_pl_Het_v1/FastBiot_v1.0_OkCor_pl9DCp
synseis_delay(1800:end)=0;
figure(5);clf
plot(real(fft(Rec2)),'r-','LineWidth',1,'DisplayName','Vy rec'); hold on;
plot(real(fft(synseis_delay)),'b.-'); hold on;

synseis_delayS=zeros(5100,1);synseis_delayS(1:2550)=flip(synseis_delay);synseis_delayS(2551:end)=synseis_delay;
Fsh=abs(fft(Rec2)); Fg = abs(fft(synseis_delay));
figure(2);clf;plot(Rec2,'r');hold on;plot(synseis_delayS,'b'); 

%figure(7);clf;plot(Fsh,'r');hold on;plot(Fg,'b');
%Fsh(180:1400)=0*Fg(180:1400); Fsh(1400:2372)=0*Fg(1400:2372);
%Fg(180:1400)=0*Fg(180:1400); Fg(1400:2372)=0*Fg(1400:2372);
figure(7);clf;semilogy(Fsh,'r');hold on;semilogy(Fg,'b');

shift =50;L_fr = 0+shift; R_fr = 2550-shift;  Atten=linspace(1,0,2550).^40; % Atten=flip ( log( linspace(1,0,2550)  )) ./log(2550);

FgR = real(fft(synseis_delay));FgI = imag(fft(synseis_delay));
FgR(L_fr:1400)=Atten(1:1401-L_fr)'.*FgR(L_fr:1400);  FgR(1400:R_fr)=flip(Atten(1: R_fr-1399)').*FgR(1400:R_fr);
FgFull = FgR + 1i*FgI;
FgINV = real(ifft(FgFull));
FgINV = smoothdata(FgINV,'gaussian',30) ; FgINV = smoothdata(FgINV,'gaussian',30) ; 

FshR = real(fft(Rec2));FshI = imag(fft(Rec2));
%FshR(L_fr:1400)=Atten(L_fr:1400)'.*FshR(L_fr:1400); FshR(1400:R_fr)=Atten(1400:R_fr)'.*FshR(1400:R_fr);
FshFull = FshR + 1i*FshI;
FshINV = real(ifft(FshFull));
FshINV = smoothdata(FshINV,'gaussian',30) ; 

figure(7);clf;plot(abs(FgR),'b');hold on;plot(abs(FshR),'r');
figure(7);clf; plot(FshINV(1:end),'r'); hold on;plot(FgINV,'b-');

EEE=linspace(1,0,2550).^50;
EEE = flip ( log( linspace(1,0,2550)  )) ./log(2550);
figure(7);clf;  plot(abs(EEE),'r');
 
%Fg = (fft(synseis_delay));
FshINVg = real(ifft(Fg)); FshINVg_s=FshINVg;  figure(7);clf; plot(FshINVg(1:1800),'b-.');
%FshINV_s(222:721)=FshINV(1:500); 
%FshINV_s(1:222)=FshINV(end-221:end);
figure(7);clf;plot(FshINV_s','r');hold on;plot(FshINVg_s,'b-.');


%% 2D plot, Vx, Vy, Vz
figure(1),clf,
Vz_plot   = Ampl*((Vz(:,:,2:end)  + Vz(:,:,1:end-1))/2);
Vz_tr = (squeeze(Vz_plot(:,fix(ny/2),:)));  %Vz_tr(:,end/2) = 10+1*Vz_tr(:,end/2);% Vz_tr(end/2,:) = 10*Vz_tr(end/2,:);
S1 = subplot(1,1,1);surf(  flip(Vz_tr')); %flip
caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.2);
colorbar, colormap jet(500), view(0,90);axis square tight; %shading interp,xlabel('m'); ylabel('m'); 
view(0,90);colorbar;shading interp;title('Vz');%caxis([-0.6; 0.6]);
colormap(S1,Red_blue_colormap);

set(gca, 'XTick', (  0  :  100  : 768) ,'fontsize',12 ); 
xticklabels({'0','1','2','3','4','5','6','7'})
xlabel('x (km)'); 
yticks([048 148 248 348 448])
yticklabels({'4','3','2','1','0'})
ylabel('z (km)'); 

pbaspect([nx nz nz])

cb = colorbar; %set(cb,'position',[0.85, 0.16, 0.04, 0.165])
title(cb,'V_{z} (m/s)');
hold on;
scatter(503,448,70,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;

figure(2),clf; plot( Vz_tr(end/2,:));
figure(2),clf; plot( Vz_tr(:,150));
%%
S1=figure(1);clf,
Vx_plot   = Ampl*((Vx(2:end,:,:)  + Vx(2:end,:,:))/2);  
Vx_tr = (squeeze(Vx_plot(:,fix(ny/2),:)));  
%Vx_tr = (squeeze(Vx_plot(:,:,2)));  
Vx_tr = flip(Vx_tr')               ; Vx_tr(346:348,233:235)=max(Vx_tr(:));Vx_tr(446:448,503:505)=max(Vx_tr(:));
S1 = subplot(1,1,1);surf(  (Vx_tr)); %flip
caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.2);
colorbar, colormap jet(500), view(0,90);axis tight; %shading interp,xlabel('m'); ylabel('m');  square
view(0,90);colorbar;shading interp;%title('Vx');%caxis([-0.6; 0.6]);
colormap(S1,Red_blue_colormap);

set(gca, 'XTick', (  0  :  100  : 768) ,'fontsize',12 ); 
xticklabels({'0','1','2','3','4','5','6','7'})
xlabel('x (km)'); 
yticks([048 148 248 348 448])
yticklabels({'4','3','2','1','0'})
ylabel('z (km)'); 

pbaspect([nx nz nx])

cb = colorbar; %set(cb,'position',[0.85, 0.16, 0.04, 0.165])
title(cb,'V_{x} (m/s)');
hold on;
scatter(503,448,70,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;

%set(cb,'YTick',-0.8:0.8:0.8);sc_ph = Lx/(nx -1);
%centX = ( fix(nx/Lx/1*8.675) - fix(nx/Lx/1*0.675))/4 ;

%set(gca, 'XTick', (  fix(nx/Lx/1*0.675)  :  centX  :  fix(1022/Lx/1*8.675)) ,'fontsize',16 );


disp(['Vx(15,15,15) = ' , num2str(Vx(15,15,15),'%50.48f') ]);
disp(['Vy(15,15,15) = ' , num2str(Vy(15,15,15),'%50.48f') ]);
disp(['Vz(15,15,15) = ' , num2str(Vz(15,15,15),'%50.48f') ]);

%% sigma_xx
figure(1),clf,
sigma_xx_tr = (squeeze(sigma_xx(fix(nx/2),:,:)));  %Vz_tr(:,end/2) = 10+1*Vz_tr(:,end/2);% Vz_tr(end/2,:) = 10*Vz_tr(end/2,:);
S1 = subplot(1,1,1);surf(  (sigma_xx_tr')); %flip
caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.1);
colorbar, colormap gray(300), view(0,90);axis equal tight; shading interp,xlabel('m'); ylabel('m');
view(0,90);colorbar;shading interp;title('sigma_xx');%caxis([-0.6; 0.6]);


figure(2),clf; plot( sigma_xx_tr(320,:));
figure(2),clf; plot( sigma_xx_tr(:,150));

%delete *.res *.inf *.dat a.out

figname = 'Vx';
fig = gcf;    fig.PaperPositionMode = 'auto';
print([figname '_' int2str(0)],'-dpng','-r600')
%% het
% name=[num2str(isave) '_0_vrho_x_11_hsav.res']; id = fopen(name); vrho_x_11_h  = fread(id,DAT); fclose(id); vrho_x_11_h  = reshape(vrho_x_11_h  ,nx-1  ,ny  ,nz);
% figure(5);  surf(vrho_x_11_h(:,:,34))
%%
%nx = 300;ny=300;nz=350;
%Vx_small = Vx(200:500,200:500,1:350);

%% 3D plot, V_total
nx=768;ny=768;nz=448;
figure(2), clf
Vx_plot  = Ampl*(Vx(2:end,:,:) + Vx(1:end-1,:,:))/2;
%Vy_plot  = (Vy(:,2:end,:) + Vy(:,1:end-1,:))/2;
%Vz_plot  = (Vz(:,:,2:end) + Vz(:,:,1:end-1))/2;
%Vx_norm  = Vx_plot + Vy_plot + Vz_plot;
Vx_norm  = flip(Vx_plot,3);
%Vx_plot = flip(Vx_plot);
%Vx_plot  = (Vx_small(2:end,:,:) + Vx_small(1:end-1,:,:))/2; 

st  =2;  % downsampling step
startx  = fix( 1     );eendx   = fix( nx     );eendz   = fix( nz     );
Vx_plot2 =  Vx_plot(startx:st:eendx ,startx:st:eendx,startx:st:eendz);
%Vy_plot2 =  Vy_plot(startx:st:eendx ,startx:st:eendx,startx:st:eendx);
%Vz_plot2 =  Vz_plot(startx:st:eendx ,startx:st:eendx,startx:st:eendx);
Vx_norm2 =  Vx_norm(startx:st:eendx ,startx:st:eendx,startx:st:eendz);
Vx_plot=0;%Vy_plot=0;Vz_plot=0;
Vx_norm=0;
Vx_plot=Vx_plot2;%Vy_plot=Vy_plot2;Vz_plot=Vz_plot2;
Vx_norm=Vx_norm2;
nx = nx/st;ny = ny/st;nz = nz/st;
Vx_norm(233/st:235/st,382/st:384/st,446/st:448/st) = max(Vx_norm(:));

clf,set(gcf,'color','white','pos',[700 200 1000 800]);
S1 = figure(2);clf; hold on;
s1 = fix( nx/2  );
s2 = fix( ny/2  );
s3 = fix( nz/10    );
s4 = fix( nz/2  );
s5 = fix( nz      );
slice(Vx_norm( 1:s2  , 1:s2  ,1:s5),[],s2,[]),shading flat
slice(Vx_norm(1:s2, :  ,1:s5),s2,[],[]),shading flat
slice(Vx_norm( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(Vx_norm(1:s2, 1:s2  , :  ),[],[],s5),shading flat
slice(Vx_norm( :  ,1:s1, :  ),[],[],s4),shading flat
slice(Vx_norm( :  , :  ,1:s4),s1,[],[]),shading flat

s7=fix( nx/10  );s8 = fix( nz/1.3  );
slice(Vx_norm( :  ,1:s7,1:s8),[],nx,[]),shading flat
slice(Vx_norm( :  ,1:s7, :  ),[],[],s8),shading flat
slice(Vx_norm( :  , :  ,1:s8),s7,[],[]),shading flat

slice(Vx_norm( :  , :  ,1:s3),ny,[],[]),shading flat
slice(Vx_norm( :  , :  ,1:s3),[],nx,s3),shading flat

s7=fix( nx/10  );s8 = fix( nz  );
slice(Vx_norm( :  ,1:s7,1:s8),[],nx,[]),shading flat
slice(Vx_norm( :  ,1:s7, :  ),[],[],s8),shading flat
slice(Vx_norm( :  , :  ,1:s8),s7,[],[]),shading flat

hold off; box on;%xlabel('x (m)'); ylabel('y (m)');  zlabel('z (m)');
axis image; view(133,38)

camlight; camproj perspective
light('position',[0.6 -1 1]);    
light('position',[0.8 -1.0 1.0]);
pos = get(gca,'position'); set(gca,'position',[0.01 pos(2), pos(3), pos(4)])
h = colorbar; caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.2);
daspect([1,1,1]);  axis square; grid on;
cb = colorbar; set(cb,'position',[0.78, 0.16, 0.04, 0.165])
title(cb,'V_{x} (m/s)');%title('V_{total} (m/s)')

grid on;colormap(S1,Red_blue_colormap);
ax = gca;ax.BoxStyle = 'full';
box on;ax.LineWidth = 2;

set(gca, 'XTick', (  0  :  100/st  : 768/st) ,'fontsize',12 ); 
xticklabels({'0','1','2','3','4','5','6','7'})
xlabel('y (km)'); 
set(gca, 'YTick', (  0  :  100/st  : 768/st) ,'fontsize',12 ); 
yticklabels({'0','1','2','3','4','5','6','7'})
ylabel('x (km)'); 
zticks([048/st 148/st 248/st 348/st 448/st])
zticklabels({'4','3','2','1','0'})
zlabel('z (km)'); 

pbaspect([nx ny nz]);hold on;
scatter3(384/st,504/st,448/st,125,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;
scatter3(384/st,234/st,348/st,85,'*','filled','MarkerEdgeColor','k') ;


Am = abs(caxis);
%Vx_plot(:,128:256,:)=0;
%%Vx_plot(:,250:end,:)=0;
isosurf = 0*1.2*1e-7 + 0.0*1e-5 + 0.5*Am(1);
is1  = isosurface(Vx_norm, isosurf);
is2  = isosurface(Vx_norm, -isosurf);
his1 = patch(is1); set(his1,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-isosurf,'Facecolor','Flat','Edgecolor','none')
alpha(his1,.5)
alpha(his2,.5)



%cb = colorbar; %set(cb,'position',[0.85, 0.16, 0.04, 0.165])
%title(cb,'V_{x} (m/s)');

%alpha 0.5
Vx_plot(:,192:end,:)=0;
%Vx_plot(:,250:end,:)=0;
isosurf = Ampl*0.085*0 + 0.1;
is1  = isosurface(Vx_plot, isosurf);
is2  = isosurface(Vx_plot, -isosurf);
his1 = patch(is1); set(his1,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-isosurf,'Facecolor','Flat','Edgecolor','none')

% is1y  = isosurface(Vy_plot, isosurf);
% is2y  = isosurface(Vy_plot, -isosurf);
% his1y = patch(is1y); set(his1y,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
% his2y = patch(is2y); set(his2y,'CData',-isosurf,'Facecolor','Flat','Edgecolor','none')
% 
% isosurfz = 0.015;
% is1z  = isosurface(Vz_plot, isosurfz);
% is2z  = isosurface(Vz_plot, -isosurfz);
% his1z = patch(is1z); set(his1z,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
% his2z = patch(is2z); set(his2z,'CData',-isosurf,'Facecolor','Flat','Edgecolor','none')



%set(cb,'YTick',-0.8:0.8:0.8);sc_ph = Lx/(nx -1);
%centX = ( fix(nx/Lx/1*8.675) - fix(nx/Lx/1*0.675))/4 ;

%set(gca, 'XTick', (  fix(nx/Lx/1*0.675)  :  centX  :  fix(1022/Lx/1*8.675)) ,'fontsize',16 );
%xticklabels({'4','2','0','-2','-4'})
%set(gca, 'YTick',(  fix(nx/Lx/1*0.675)  :  centX  :  fix(1022/Lx/1*8.675)) ,'fontsize',16 );
%yticklabels({'4','2','0','-2','-4'})
%set(gca, 'ZTick', (  fix(nx/Lx/1*0.675)  :  centX  :  fix(1022/Lx/1*8.675)) ,'fontsize',16 );
%zticklabels({'-4','-2','0','2','4'})


% figname = 'Vt_total';
% fig = gcf;    fig.PaperPositionMode = 'auto';
% print([figname '_' int2str(0)],'-dpng','-r600')
% %% delete the data
% End_of_the_code = 1;
% delete *.res *.inf *.dat a.out
%% matlab
%Vx(15,15,15) = 0.030353836423165941882373886073764879256486892700
%Vy(15,15,15) = 0.025353591625612705712233818644563143607228994370
%Vz(15,15,15) = 0.055462739809972953775041304425030830316245555878
%% CUDA
% Vx(15,15,15) = 0.030353836423166000862972069285206089261919260025
% Vy(15,15,15) = 0.025353591625612736937256386227090843021869659424
% Vz(15,15,15) = 0.055462739809972828874951034094920032657682895660