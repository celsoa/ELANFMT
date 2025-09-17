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
%DAT = 'double';
DAT = 'single';
ScaleT = 1;
% NBX is the input parameter to GPU
NBX = ScaleT*24;%12
NBY = ScaleT*8;
NBZ = ScaleT*7;
nt  = 8500 +   0*3550; % number of iterations

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
Sc_dx =2;
Lx     = 2*7680 + 0*9.35/spatial_scale; % m
Lx     = 2*11520 + 0*9.35/spatial_scale; % m
Lx     = 2*15360 + 0*9.35/spatial_scale; % m
Ly     = 2*5120 + 0*9.35/spatial_scale; % m %(1330-180)*2
Lz     = Sc_dx*4480 + 0*9.35/spatial_scale; % m
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
%background=layer 2
rho1      = 2500;          % density
K1        = 1.6*15*1e9;    % Bulk  modulus [GPa]
G1        = 12*1e9;        % Shear modulus [GPa]
c11       = (K1 + 4/3*G1) ;
c13       = (K1 - 2/3*G1)  ;
c66       = G1            ;
Vp1 = sqrt( (K1 + 4/3*G1)./rho1 );
Vs1 = sqrt( ( G1)./rho1 );
vrho_x_11 = 1./rho1;

layer1dz = 200/2;

% layer 1
rho2      = 2000;      % density
K2        = 0.5*15*1e9;    % Bulk  modulus [GPa]
G2        = 6*1e9;    % Shear modulus [GPa]
rho_n     = rho1         ;
c11_L2       = (K2 + 4/3*G2) ;
c13_L2       = (K2 - 2/3*G2)  ;
vrho2_x_11 = 1./rho2;

Vp2 = sqrt( (K2 + 4/3*G2)./rho2 );
Vs2 = sqrt( ( G2)./rho2 );

dt_check       = 1./sqrt(1./dx.^2 + 1./dy.^2 + 1./dz.^2) .* 1/max(Vp1,Vp2) ;

rho =rho1;
dt  = 0.001 / ScaleT *1;% *Sc_dx;
dt  = 0.0008;
dt  = 0.00125;

c11u=c11; c33u=c11; c13u=c13; c12u=c13; c23u=c13; c44u=G1; c55u=G1; c66u=G1;
%%
%vrho_x_11 = 1./rho;
vrho_y_11 = 1./rho;
vrho_z_11 = 1./rho;

vrho_x_12 = rho_fluid./delta1;
vrho_y_12 = rho_fluid./delta2;
vrho_z_12 = rho_fluid./delta3;

vrho_x_22 = rho./delta1;
vrho_y_22 = rho./delta2;
vrho_z_22 = rho./delta3;

%%          km   Vs         Vp    Dens
Layer1=[   3    3.237      5.6    2.14   100  225  ];
Layer2=[(3+7)    3.457      5.98   2.56   100  225   ];
Layer3=[(3+7+10)  3.48       6.02   2.87   100  225   ];
Layer4=[(3+7+10+10)   3.798      6.57   3.00   100  225   ];
Layer5=[(3+7+10+10+5)    4.41       7.63   3.00   100  225   ];
Layer6=[   (3+7+10+10+5+10)    4.5140     7.81   3.29   100  225   ];
Layer7=[   (3+7+10+10+5+10+5)    4.6530     8.05   3.29   100  225   ];
Layer8=[   200  4.711      8.15   3.29   100  225   ];
Layer9=[   0    4.711      8.15   3.29   100  225  ];

VpL1 = Layer1(3)*1e3; VpL2 = Layer2(3)*1e3; VpL3 = Layer3(3)*1e3;VpL7 = Layer7(3)*1e3;
dt_check       = 1./sqrt(1./dx.^2 + 1./dy.^2 + 1./dz.^2) .* 1/max(VpL1,VpL3) ;

%Layer1=[ 2.0     1.7321    2.7839    2.0    300     600  ];
%Layer2=[6.1    2.190890    4.00    2.5    300     600  ];

Scale_Zdx = ceil(1000/dz);
layer1dz  = Layer1(1)*Scale_Zdx;
rho_L1     = Layer1(4)*1000;
c11_L1     = Layer1(3)^2 * rho_L1/1000*1e9;
c44_L1     = Layer1(2)^2 * rho_L1/1000*1e9;
K_L1        = (c11_L1 - 4/3*c44_L1);
c12_L1     =  K_L1 - 2/3*c44_L1 ;
vrho_x_L1 = 1./rho_L1;

layer2dz  = Layer2(1)*Scale_Zdx;
rho_L2     = Layer2(4)*1000;
c11_L2     = Layer2(3)^2 * rho_L2/1000*1e9;
c44_L2     = Layer2(2)^2 * rho_L2/1000*1e9;
K_L2       = (c11_L2 - 4/3*c44_L2);
c12_L2     =  K_L2 - 2/3*c44_L2 ;
vrho_x_L2 = 1./rho_L2;

layer3dz  = Layer3(1)*Scale_Zdx;
rho_L3     = Layer3(4)*1000;
c11_L3    = Layer3(3)^2 * rho_L3/1000*1e9;
c44_L3     = Layer3(2)^2 * rho_L3/1000*1e9;
K_L3       = (c11_L3  -4/3*c44_L3);
c12_L3     =  K_L3 - 2/3*c44_L3 ;
vrho_x_L3 = 1./rho_L3;

layer4dz  = Layer4(1)*Scale_Zdx;
rho_L4     = Layer4(4)*1000;
c11_L4    = Layer4(3)^2 * rho_L4/1000*1e9;
c44_L4     = Layer4(2)^2 * rho_L4/1000*1e9;
K_L4       = (c11_L4  -4/3*c44_L4);
c12_L4     =  K_L4 - 2/3*c44_L4 ;
vrho_x_L4 = 1./rho_L3;

layer5dz  = Layer5(1)*Scale_Zdx;
rho_L5     = Layer5(4)*1000;
c11_L5    = Layer5(3)^2 * rho_L5/1000*1e9;
c44_L5     = Layer5(2)^2 * rho_L5/1000*1e9;
K_L5       = (c11_L5  -4/3*c44_L5);
c12_L5     =  K_L5 - 2/3*c44_L5 ;
vrho_x_L5 = 1./rho_L5;

layer6dz  = Layer6(1)*Scale_Zdx;
rho_L6     = Layer6(4)*1000;
c11_L6    = Layer6(3)^2 * rho_L6/1000*1e9;
c44_L6     = Layer6(2)^2 * rho_L6/1000*1e9;
K_L6       = (c11_L6  -4/3*c44_L6);
c12_L6     =  K_L6 - 2/3*c44_L6 ;
vrho_x_L6 = 1./rho_L6;

layer7dz  = Layer7(1)*Scale_Zdx;
rho_L7     = Layer7(4)*1000;
c11_L7    = Layer7(3)^2 * rho_L7/1000*1e9;
c44_L7     = Layer7(2)^2 * rho_L7/1000*1e9;
K_L7       = (c11_L7  -4/3*c44_L7);
c12_L7     =  K_L7 - 2/3*c44_L7 ;
vrho_x_L7 = 1./rho_L7;

layer8dz  = Layer8(1)*Scale_Zdx;
rho_L8     = Layer8(4)*1000;
c11_L8    = Layer8(3)^2 * rho_L8/1000*1e9;
c44_L8     = Layer8(2)^2 * rho_L8/1000*1e9;
K_L8       = (c11_L8  -4/3*c44_L8);
c12_L8     =  K_L8 - 2/3*c44_L8 ;
vrho_x_L8 = 1./rho_L8;
%%
[x, y, z]        = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2,-Lz/2:dz:Lz/2);
vrho_x_11_h = vrho_x_L2 + 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2);
vrho_x_11_h(:,:,1:layer1dz)  =  vrho_x_L1; 
%vrho_x_11_h(:,:,layer1dz:layer2dz)  = vrho_x_L2;
% vrho_x_11_h(:,:,layer2dz:layer3dz)  =vrho_x_L3;
% vrho_x_11_h(:,:,layer3dz:layer4dz)  =vrho_x_L4;
% vrho_x_11_h(:,:,layer4dz:layer5dz)  =vrho_x_L5;
% vrho_x_11_h(:,:,layer5dz:layer6dz)  =vrho_x_L6;
% vrho_x_11_h(:,:,layer6dz:layer7dz)  =vrho_x_L7;

%vrho_x_11_h(:,:,1)  = 0.00001;
vrho_y_11_h = vrho_x_11_h; 
vrho_z_11_h = vrho_x_11_h;

vrho_x_11_hs = (vrho_x_11_h(1:end-1,:,:)+vrho_x_11_h(2:end,:,:))./2; 
vrho_x_11_h = zeros(nx+1,ny,nz);
vrho_x_11_h(2:end-1,:,:) = vrho_x_11_hs;
fid                = fopen('vrho_x_11_hc.dat','wb'); fwrite(fid,vrho_x_11_h(:),DAT); fclose(fid);

vrho_y_11_hs = (vrho_y_11_h(:,1:end-1,:)+vrho_y_11_h(:,2:end,:))./2;
vrho_y_11_h = zeros(nx,ny+1,nz);
vrho_y_11_h(:,2:end-1,:) = vrho_y_11_hs;
fid                = fopen('vrho_y_11_hc.dat','wb'); fwrite(fid,vrho_y_11_h(:),DAT); fclose(fid);

vrho_z_11_hs = (vrho_z_11_h(:,:,1:end-1)+vrho_z_11_h(:,:,2:end))./2;
vrho_z_11_h = zeros(nx,ny,nz+1);
vrho_z_11_h(:,:,2:end-1) = vrho_z_11_hs;
fid                = fopen('vrho_z_11_hc.dat','wb'); fwrite(fid,vrho_z_11_h(:),DAT); fclose(fid);

c11u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c11_L2;
c11u_het(:,:,1:layer1dz)  =  c11_L1;                       
%c11u_het(:,:,layer1dz:layer2dz) = c11_L2;
% c11u_het(:,:,layer2dz:layer3dz) = c11_L3;
% c11u_het(:,:,layer3dz:layer4dz) = c11_L4;
% c11u_het(:,:,layer4dz:layer5dz) = c11_L5;
% c11u_het(:,:,layer5dz:layer6dz) = c11_L6;
% c11u_het(:,:,layer6dz:layer7dz) = c11_L7;
%c11u_het(:,:,1) = 0;
fid           = fopen('c11u_het.dat','wb'); fwrite(fid,c11u_het(:),DAT); fclose(fid);    

c12u_het = 0.0.*exp(-(x/lamx).^2-(y/lamy).^2 - (z/lamz).^2) + c12_L2;
c12u_het(:,:,1:layer1dz)  =  c12_L1 ; 
%c12u_het(:,:,layer1dz:layer2dz) = c12_L2;
% c12u_het(:,:,layer2dz:layer3dz) = c12_L3;
% c12u_het(:,:,layer3dz:layer4dz) = c12_L4;
% c12u_het(:,:,layer4dz:layer5dz) = c12_L5;
% c12u_het(:,:,layer5dz:layer6dz) = c12_L6;
% c12u_het(:,:,layer6dz:layer7dz) = c12_L7;
%c12u_het(:,:,1) = 0;
fid           = fopen('c12u_het.dat','wb'); fwrite(fid,c12u_het(:),DAT); fclose(fid);

G_het                = zeros(nx,ny,nz) + c44_L2;
G_het(:,:,1:layer1dz)  =  c44_L1; 
%G_het(:,:,layer1dz:layer2dz) = c44_L2;
% G_het(:,:,layer2dz:layer3dz) = c44_L3;
% G_het(:,:,layer3dz:layer4dz) = c44_L4;
% G_het(:,:,layer4dz:layer5dz) = c44_L5;
% G_het(:,:,layer5dz:layer6dz) = c44_L6;
% G_het(:,:,layer6dz:layer7dz) = c44_L7;
%G_het(:,:,1) = 0.000001;
G_hetxy             = 4./( 1./G_het(1:end-1,1:end-1,:) + 1./G_het(2:end,1:end-1,:) + 1./G_het(1:end-1,2:end,:) + 1./G_het(2:end,2:end,:)    );
c44u_hetxy         = zeros(nx+1,ny+1,nz);
c44u_hetxy(2:end-1,2:end-1,:) = G_hetxy;
fid           = fopen('c44u_hetxy.dat','wb'); fwrite(fid,c44u_hetxy(:),DAT); fclose(fid);

G_hetxz             = 4./( 1./G_het(1:end-1,:,1:end-1) + 1./G_het(2:end,:,1:end-1) + 1./G_het(1:end-1,:,2:end) + 1./G_het(2:end,:,2:end)    );
c44u_hetxz         = zeros(nx+1,ny,nz+1);
c44u_hetxz(2:end-1,:,2:end-1) = G_hetxz;
fid           = fopen('c44u_hetxz.dat','wb'); fwrite(fid,c44u_hetxz(:),DAT); fclose(fid);

G_hetyz             = 4./( 1./G_het(:,1:end-1,1:end-1) + 1./G_het(:,2:end,1:end-1) + 1./G_het(:,1:end-1,2:end) + 1./G_het(:,2:end,2:end)    );
c44u_hetyz         = zeros(nx,ny+1,nz+1);
c44u_hetyz(:,2:end-1,2:end-1) = G_hetyz;
fid           = fopen('c44u_hetyz.dat','wb'); fwrite(fid,c44u_hetyz(:),DAT); fclose(fid);

%% Attenuation
abs_thick = min(floor(0.2*nx), floor(0.2*nx));         % thicknes of the layer
abs_thick = 120;%128
abs_rate = 0.6*0.16/abs_thick;      % decay rate

lmargin = [abs_thick abs_thick abs_thick];
rmargin = lmargin;
marginY = 210+0*floor(0.42*ny);
weights = ones(nx,ny,nz);
for iy = 1:ny
    for ix = 1:nx
        for iz = 1:nz
        i = 0;
        j = 0;
        k = 0;
        if (ix < lmargin(1) + 1)
            i = lmargin(1) + 1 - ix;
        end
        if (iy < marginY + 1)
            k = marginY + 1 - iy;
        end
        %if (iz < lmargin(3) + 1)
        %    j = lmargin(3) + 1 - iz;
        %end
        if (nx - rmargin(1) < ix)
            i = ix - nx + rmargin(1);
        end
        if (ny - marginY < iy)
            k = iy - ny + marginY;
        end
        if (nz - rmargin(3) < iz)
            j = iz - nz + rmargin(3);
        end
        if (i == 0 && j == 0 && k == 0)
            continue
        end
        rr = abs_rate * abs_rate * double(i*i + j*j + k*k );
        weights(ix,iy,iz) = exp(-rr);
        end
    end
end
%plot(weights(:,150))
%weights(1:250,:,:) = 
figure(1)
Pl = squeeze (weights(:,:,150));
mesh(Pl);view(0,90),colorbar
fid           = fopen('weights.dat','wb'); fwrite(fid,weights(:),DAT); fclose(fid);
%%
pa1           = [dx dy dz dt Lx Ly Lz lamx lamy lamz];
pa2           = [c11u c33u c13u c12u c23u c44u c55u c66u alpha1 alpha2 alpha3 M1 c22u];
%pa3           = [ vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22];
pa3           = [vrho_x_11 vrho_y_11 vrho_z_11 vrho_x_12 vrho_y_12 vrho_z_12 vrho_x_22 vrho_y_22 vrho_z_22 eta_k1_av eta_k2_av eta_k3_av ];
fid           = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid           = fopen('pa2.dat','wb'); fwrite(fid,pa2(:),DAT); fclose(fid);
fid           = fopen('pa3.dat','wb'); fwrite(fid,pa3(:),DAT); fclose(fid);
%%
% dur = 800/2;
% srctimefun = 1e0*load('srctimefunL.txt');
% figure(6);clf; plot(srctimefun); hold on;
% [x, y]   = ndgrid(-dur*20:dx*1:dur*20,-Ly/2:dy:Ly/2);
% lamx = 1000;
% init = 2.5*1e-3*exp(-(x(1:end,1)/lamx/2).^2 );
% plot(init); hold on;
% SourceGauss = zeros(1,nt); SourceGauss(1:2*dur-3) = -1e9*diff(diff(init')); 
% SourceGauss = SourceGauss/max(SourceGauss(:))*2.5*1e-3;
% figure(6), plot(SourceGauss);
% Src = SourceGauss';
% fid           = fopen('Src.dat','wb'); fwrite(fid,Src(:),DAT); fclose(fid);
% 
% name=[num2str(0) '_0_SrcSV.res']; id = fopen(name); SrcSV   = fread(id,DAT); fclose(id); 
% SrcSV   = reshape(SrcSV  ,nt  ,1  ,1);
% Source8500 = zeros(8500,1);Source8500(1:4500)=SrcSV;
% figure(6), clf;plot(Source8500);
% fid           = fopen('Src.dat','wb'); fwrite(fid,Source8500(:),DAT); fclose(fid);

%Src = Src./max(Src(:));
% fid  = fopen('SrcLR.txt','w'); fprintf(fid,'%6.6f\n',Src);  fclose(fid);
 dur = 200;

% 
% srctimefun = load('srctimefun.txt');
% srctimefun = 3.3*10e10*interp1(1:101,srctimefun,1:0.5:100);%3.3e20*
% S = size(srctimefun);S = S(2);
% SourceSharp = zeros(1,nt); SourceSharp(1:S(1)+0) = srctimefun;
% 
% [x, y]   = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2);
% lamx = 350;
% init = 7*2e9*exp(-(x(1:end,1)/lamx/Sc_dx).^2 );
% SourceGauss = zeros(1,nt); SourceGauss(1:nx-99) = init(100:end);
% 
% lamx = lamx;
% init = 3.3*2e9*exp(-(x(1:end,1)/lamx/Sc_dx).^2 );
% SourceGauss = zeros(1,nt); SourceGauss(1:nx-101) = init(102:end); figure(2), plot(SourceGauss)
% %SourceSharp(180:260) = max(SourceSharp(180:260) ,SourceGauss(180:260));
% GGG = SourceGauss(1:nt);
% %SourceSharp(:) = max(SourceSharp(:) ,GGG(:));
% 
% SourceGauss = zeros(1,nt); SourceGauss(1:nx-dur+1) = init(dur:end);
% %SourceSharp(1:round(140*dur/200)) = max(SourceSharp(1:round(140*dur/200)) ,SourceGauss(1:round(140*dur/200)) );
% %SourceSharp(round(300*dur/200):end) = max(SourceSharp(round(300*dur/200):end)  ,SourceGauss(round(300*dur/200):end)  );
% %SourceSharp(round(180*dur/200):round(260*dur/200)) = max(SourceSharp(round(180*dur/200):round(260*dur/200)) ,SourceGauss(round(180*dur/200):round(260*dur/200)) );
% for ii=1:4
%   SourceSharp = smoothdata(SourceSharp,'gaussian',15) ;
%   SourceSharp = smoothdata(SourceSharp,'gaussian',15) ;
% end
% 
% t        = (0:nt-1)*dt;
% 
% max_SourceGauss = max(SourceGauss(:));
% 
% Src = SourceGauss';
% figure(6);clf;plot(Src,'r');hold on;plot(SourceSharp,'b--');
% figure(2);clf;plot(diff(SourceSharp,2),'r');hold on;plot(diff(SourceGauss,2),'b');
% 
% SrcNew = Src(1:2:nt);
% 
% 
% %Src = 0*SourceGauss'; Src(1:nt/2) = SrcNew;
% figure(3); plot(Src,'b--');
% % 
% zz=1;
% % Fsh=abs(fft(SourceSharp)); Fg = abs(fft(SourceGauss));
% % figure(1);clf;plot(Fsh,'r');hold on;plot(Fg,'b');
% % Fsh(33:500)=Fg(33:500); Fsh(1000:2517)=Fg(1000:2517); figure(1);clf;plot(Fsh,'r');hold on;plot(Fg,'b');
% % FshINV = real(ifft(Fsh)); FshINV_s=FshINV*0; 
% % FshINV_s(222:721)=FshINV(1:500); 
% % FshINV_s(1:222)=FshINV(end-221:end);
% % FshINV_s(1:140) = max(FshINV_s(1:140) ,SourceGauss(1:140)' ); FshINV_s(1:100) = SourceGauss(1:100)' ;
% % FshINV_s(300:end) = max(FshINV_s(300:end)  ,SourceGauss(300:end)'  );FshINV_s(350:end) = SourceGauss(350:end)';
% % figure(1);clf;plot(FshINV_s','r');hold on;plot(SourceGauss,'b');
% % %Src = FshINV_s;
% 
% %fid           = fopen('Src.dat','wb'); fwrite(fid,Src(:),'double'); fclose(fid);
%% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
code_name    = 'FastWaveED_GPU3D_v1_23kmst'; %--compiler-options
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
%name=[num2str(isave) '_0_Vx.res']; id = fopen(name); Vx  = fread(id,DAT); fclose(id); Vx  = reshape(Vx  ,nx+1,ny  ,nz  );
%name=[num2str(isave) '_0_Vy.res']; id = fopen(name); Vy  = fread(id,DAT); fclose(id); Vy  = reshape(Vy  ,nx  ,ny+1,nz  );
%name=[num2str(isave) '_0_Vz.res']; id = fopen(name); Vz  = fread(id,DAT); fclose(id); Vz  = reshape(Vz  ,nx  ,ny  ,nz+1);
name=[num2str(isave) '_0_Src.res']; id = fopen(name); Src   = fread(id,DAT); fclose(id); Src   = reshape(Src  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec.res']; id = fopen(name); Rec   = fread(id,DAT); fclose(id); Rec  = reshape(Rec  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_x.res']; id = fopen(name); Rec_x   = fread(id,DAT); fclose(id); Rec_x  = reshape(Rec_x  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y.res']; id = fopen(name); Rec_y   = fread(id,DAT); fclose(id); Rec_y  = reshape(Rec_y  ,nt  ,1  ,1);
name=[num2str(isave) '_0_Rec_y2.res']; id = fopen(name); Rec_y2   = fread(id,DAT); fclose(id); Rec_y2  = reshape(Rec_y2  ,nt  ,1  ,1);

%name=[num2str(isave) '_0_Rec1.res']; id = fopen(name); Rec1   = fread(id,DAT); fclose(id); Rec1  = reshape(Rec1  ,nt*0+1600  ,1  ,1);
%name=[num2str(isave) '_0_Rec_x1.res']; id = fopen(name); Rec_x1   = fread(id,DAT); fclose(id); Rec_x1  = reshape(Rec_x1  ,nt*0+1600  ,1  ,1);
%name=[num2str(isave) '_0_Rec_y1.res']; id = fopen(name); Rec_y1   = fread(id,DAT); fclose(id); Rec_y1  = reshape(Rec_y1  ,nt*0+1600  ,1  ,1);

%name=[num2str(isave) '_0_sigma_xx.res']; id = fopen(name); sigma_xx  = fread(id,DAT); fclose(id); sigma_xx  = reshape(sigma_xx  ,nx,ny  ,nz  );
% name=[num2str(isave) '_0_xx.res'];  id = fopen(name); XX   = fread(id,DAT); fclose(id); XX   = reshape(XX  ,nx  ,ny  ,nz  );
% name=[num2str(isave) '_0_yy.res'];  id = fopen(name); YY   = fread(id,DAT); fclose(id); YY   = reshape(YY  ,nx  ,ny  ,nz  );
% name=[num2str(isave) '_0_zz.res'];  id = fopen(name); ZZ   = fread(id,DAT); fclose(id); ZZ   = reshape(ZZ  ,nx  ,ny  ,nz  );
%%
figure(2),clf,
plot(Src); title('Source GPU');
%% m0=3.3e20; mt=[0,0,1; 0,0,0.8; 1,0.8,0]; dura=0.2; az=0.0; outnm ='synseis'; nam='vsimple_1.0/2.70.grn.';
figure(4),clf,
Max_rec = 2048*2;
isyn_c_nameZ = sprintf('23synseis.z.txt');isyn_cZ = load(isyn_c_nameZ);
delay = 1440;synseis_delay = zeros(1,nt);synseis_delay(delay+1:Max_rec+delay)=isyn_cZ(1: [Max_rec]) ;

isyn_c_nameX = sprintf('23synseis.r.txt');isyn_cX = load(isyn_c_nameX);
synseis_delayX = zeros(1,nt);synseis_delayX(delay+1:Max_rec+delay)=isyn_cX(1: [Max_rec]) ;

isyn_c_nameY = sprintf('23synseis.t.txt');isyn_cY = load(isyn_c_nameY);
synseis_delayY = zeros(1,nt);synseis_delayY(delay+1:Max_rec+delay)=isyn_cY(1: [Max_rec]) ;

Ampl = 1e9 *4.9 *0.5   *0.25*1.5*1.65*0.75 ;
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
end_minus = 0;
start_Zhu = 1000*0 +101; end_Zhu = 1;
size_x = size(synseis_delay(start_Zhu:end-end_Zhu));

start_num = 1000*0 + 1; start_num2 = 1820;
x_timeGPU =( start_num:1:(nt-end_minus) )'.*dt;
x_timeZHU =( start_num:1:(size_x(2) - 0*end_minus) )'.*dt;
clf;
%%

subplot(1,3,1); %clf
ZhuF=synseis_delay(start_Zhu:end-end_Zhu);
GPUF=1.5/3/1e-9*Rec2((start_num+start_num2):end-end_minus)*0.3;%1.002;
 
 flo = 1/25;%0.065;      % Hz
fhi = 1/12.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:6680)=GPUF(1:6680);GPUF=signal;
signal = zeros(150000,1);signal(1:6680)=ZhuF(1:6680)';ZhuF=signal;
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

plot( Tsignal(1+1000:end)/33.3407,yZ(1:end-1000),'r-','LineWidth',1.5); hold on;
plot( Tsignal(1+0:end)/33.3407,yG(1:end-0),'b-.','LineWidth',1.5); title('Vertical'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');% ylim([-5e-3 5e-3]); xlim([1 5.5]);   %Rec2(1000:end-end_minus)
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-3.5e-4 3.5e-4]);
 
%clf 
subplot(1,3,2);% clf
ZhuF=(synseis_delayX(start_Zhu:end-end_Zhu));
GPUF=1.2/3/1e-9*(RecX((start_num+start_num2):end-end_minus))*0.085;%1.3;
 
signal = zeros(150000,1);signal(1:6680)=GPUF(1:6680);GPUF=signal;
signal = zeros(150000,1);signal(1:6680)=ZhuF(1:6680)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
 flo = 1/25;%0.065;      % Hz
fhi = 1/12.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz
  flo = flo/(fs_Hz/2);
 fhi = fhi/(fs_Hz/2);
 [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal/33.3407, yZ,'r-','LineWidth',1.5); hold on;
plot(Tsignal(1+5900:end)/33.3407, yG(1:end-5900),'b-.','LineWidth',1.5); title('Radial'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D');%ylim([-2e-3 2e-3]); xlim([1 5.5]);  
 xlabel(' Time (s)'); ylabel('v (m/s)');  legend boxoff;ylim([-1.3*1e-4 1.3*1e-4]);
 ylim([-3.5e-4 3.5e-4]);

%clf
subplot(1,3,3); %clf
ZhuF=synseis_delayY(start_Zhu:end-end_Zhu);
GPUF=12/3/1e-7*RecY((start_num+start_num2):end-end_minus-0*20)*0.155;
 
 flo = 1/25;%0.065;      % Hz
fhi = 1/12.5;%0.11;      % Hz
fs_Hz = 1/dt;% 1.0/dt; % sampling rate, Hz

signal = zeros(150000,1);signal(1:6680)=GPUF(1:6680);GPUF=signal;
signal = zeros(150000,1);signal(1:6680)=ZhuF(1:6680)';ZhuF=signal;
Tsignal = 1*dt:1*dt:150000*dt;
flo = flo/(fs_Hz/2);
fhi = fhi/(fs_Hz/2);
 
  [z, p, k] = butter(4,[flo fhi],'bandpass');
sos = zp2sos(z,p,k);  % convert to zero-pole-gain filter parameter. NEEDED?
yG = sosfilt(sos, GPUF);    % apply filter
yZ = sosfilt(sos, ZhuF );

plot(Tsignal(1+6500:end)/33.3407,yZ(1:end-6500),'r-','LineWidth',1.5); hold on;
plot(Tsignal/33.3407,yG,'b-.','LineWidth',1.5); title('Transverse'); 
 legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); %ylim([-1e-2 1e-2]); xlim([1 5.5]);   
 xlabel(' Time (s)'); ylabel('v (m/s)');  ylim([-5*1e-5 5*1e-5]);
legend boxoff
ylim([-3.5e-4 3.5e-4]);
%%
if 4>3; return; end
%%

subplot(1,3,1); plot(x_timeZHU,synseis_delay(start_Zhu:end-end_Zhu),'r-','LineWidth',3.0); hold on;
plot(x_timeGPU(1:end-start_num2),+2/4.8/1e-7*Rec2((start_num+start_num2):end-end_minus),'b-.','LineWidth',2.0); title('Vertical'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-2 2]); xlim([1 5.5]);   %Rec2(1000:end-end_minus)
%hold on; plot(Ampl.*Rec1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

subplot(1,3,2); plot(x_timeZHU,(synseis_delayX(start_Zhu:end-end_Zhu)),'r-','LineWidth',3.0); hold on; 
plot(x_timeGPU(1:end-start_num2),-2/4.8/1e-7*(RecX((start_num+start_num2):end-end_minus)),'b-.','LineWidth',2.0); title('Radial'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-3 3]); xlim([1 5.5]);  
%hold on; plot((-Ampl.*Rec_x1(1000:end-end_minus)),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); legend boxoff

subplot(1,3,3); plot(x_timeZHU,synseis_delayY(start_Zhu:end-end_Zhu),'r-','LineWidth',3.0); hold on; 
plot(x_timeGPU(1+50*0:end-start_num2),-3/4.8/1e-7*RecY((start_num+start_num2):end-end_minus-50*0),'b-.','LineWidth',2.0); title('Transverse'); 
legend('Zhu and Rivera (2002)', 'Present study, FastWave GPU3D'); ylim([-3 3]); xlim([1 5.5]);   
%hold on; plot(-Ampl.*Rec_y1(1000:end-end_minus),'g-','LineWidth',2.0);
xlabel(' Time (sec)'); %ylabel('  1/Q','fontsize',18); 
legend boxoff

zzz = 1;
%delete *.res *.inf  a.exe a.exp a.lib
 

%% 2D plot, Vx, Vy, Vz
% figure(1),clf,
% Vz_plot   = ((Vz(:,:,2:end)  + Vz(:,:,1:end-1))/2);
% Vz_tr = (squeeze(Vz_plot(fix(nx/2),:,:)));  %Vz_tr(:,end/2) = 10+1*Vz_tr(:,end/2);% Vz_tr(end/2,:) = 10*Vz_tr(end/2,:);
% S1 = subplot(1,1,1);surf(  (Vz_tr')); %flip
% caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.8);
% colorbar, colormap jet(500), view(0,90);axis square tight; %shading interp,xlabel('m'); ylabel('m'); 
% view(0,90);colorbar;shading interp;title('Vz');%caxis([-0.6; 0.6]);
% 
% figure(2),clf; plot( Vz_tr(end/2,:));
% figure(2),clf; plot( Vz_tr(:,150));
%%
S1=figure(1);clf,
Vx_plot   = Ampl*((Vx(2:end,:,:)  + Vx(2:end,:,:))/2);  
Vx_tr = (squeeze(Vx_plot(:,fix(ny/2),:)));  
Vx_tr = flip(Vx_tr')               ; Vx_tr(248:250,180:182)=max(Vx_tr(:));Vx_tr(446:448,630:632)=max(Vx_tr(:));
S1 = subplot(1,1,1);surf(  (Vx_tr)); %flip
caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.6);
colorbar, colormap jet(500), view(0,90);axis tight; %shading interp,xlabel('m'); ylabel('m');  square
view(0,90);colorbar;shading interp;%title('Vx');%caxis([-0.6; 0.6]);
colormap(S1,Red_blue_colormap);
pbaspect([nx ny nz]);

set(gca, 'XTick', (  0  :  100  : 768) ,'fontsize',12 ); 
xticklabels({'0','2','4','6','8','10','12','14'})
xlabel('x (km)'); 
yticks([048 148 248 348 448])
yticklabels({'8','6','4','2','0'})
ylabel('z (km)'); 

pbaspect([nx nz nx])

cb = colorbar; %set(cb,'position',[0.85, 0.16, 0.04, 0.165])
title(cb,'V_{x} (m/s)');
st = 1;hold on;
scatter(630/st,448/st,70,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;
%scatter(180/st,248/st,185,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;
%%
% S1=figure(1);clf,
% Vy_plot   = Ampl*((Vy(:,2:end,:)  + Vy(:,2:end,:))/2);  
% Vy_tr = (squeeze(Vx_plot(:,:,2)));  
% Vy_tr = flip(Vy_tr')               ; %Vx_tr(446:448,233:235)=max(Vx_tr(:));Vx_tr(446:448,503:505)=max(Vx_tr(:));
% S1 = subplot(1,1,1);surf(  (Vy_tr)); %flip
% caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.2);
% colorbar, colormap jet(500), view(0,90);axis tight; %shading interp,xlabel('m'); ylabel('m');  square
% view(0,90);colorbar;shading interp;%title('Vx');%caxis([-0.6; 0.6]);
% colormap(S1,Red_blue_colormap);
% pbaspect([nx ny nz]);
% 
% 
% disp(['Vx(15,15,15) = ' , num2str(Vx(15,15,15),'%50.48f') ]);
% disp(['Vy(15,15,15) = ' , num2str(Vy(15,15,15),'%50.48f') ]);
% disp(['Vz(15,15,15) = ' , num2str(Vz(15,15,15),'%50.48f') ]);

%% sigma_xx
% figure(1),clf,
% sigma_xx_tr = (squeeze(sigma_xx(fix(nx/2),:,:)));  %Vz_tr(:,end/2) = 10+1*Vz_tr(:,end/2);% Vz_tr(end/2,:) = 10*Vz_tr(end/2,:);
% S1 = subplot(1,1,1);surf(  (sigma_xx_tr')); %flip
% caxis([-max(abs(caxis)); max(abs(caxis))]);  caxis(caxis*0.1);
% colorbar, colormap gray(300), view(0,90);axis equal tight; shading interp,xlabel('m'); ylabel('m');
% view(0,90);colorbar;shading interp;title('sigma_xx');%caxis([-0.6; 0.6]);
% 
% figure(2),clf; plot( sigma_xx_tr(320,:));
% figure(2),clf; plot( sigma_xx_tr(:,150));
% 
% %delete *.res *.inf *.dat a.out
% 
% figname = 'Vx';
% fig = gcf;    fig.PaperPositionMode = 'auto';
% print([figname '_' int2str(0)],'-dpng','-r600')
 
%% 3D plot, V_total
nx=768;ny=512;nz=448;
figure(2), clf
Vx_plot  = Ampl*(Vx(2:end,:,:) + Vx(1:end-1,:,:))/2;
%Vy_plot  = (Vy(:,2:end,:) + Vy(:,1:end-1,:))/2;
%Vz_plot  = (Vz(:,:,2:end) + Vz(:,:,1:end-1))/2;
%Vx_norm  = Vx_plot + Vy_plot + Vz_plot;
Vx_norm  = flip(Vx_plot,3);
%Vx_plot = flip(Vx_plot);
%Vx_plot  = (Vx_small(2:end,:,:) + Vx_small(1:end-1,:,:))/2; 

st  =2;  % downsampling step
startx  = fix( 1     );eendx   = fix( nx     );eendy   = fix( ny     );eendz   = fix( nz     );
Vx_plot2 =  Vx_plot(startx:st:eendx ,startx:st:eendy,startx:st:eendz);
%Vy_plot2 =  Vy_plot(startx:st:eendx ,startx:st:eendx,startx:st:eendx);
%Vz_plot2 =  Vz_plot(startx:st:eendx ,startx:st:eendx,startx:st:eendx);
Vx_norm2 =  Vx_norm(startx:st:eendx ,startx:st:eendy,startx:st:eendz);
Vx_plot=0;%Vy_plot=0;Vz_plot=0;
Vx_norm=0;
Vx_plot=Vx_plot2;%Vy_plot=Vy_plot2;Vz_plot=Vz_plot2;
Vx_norm=Vx_norm2;
nx = nx/st;ny = ny/st;nz = nz/st;
%Vx_norm(233/st:235/st,382/st:384/st,446/st:448/st) = max(Vx_norm(:));

clf,set(gcf,'color','white','pos',[700 200 1000 800]);
S1 = figure(2);clf; hold on;
s1 = fix( ny/2  );
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

set(gca, 'XTick', (  0  :  100/st  : 512/st) ,'fontsize',12 ); 
xticklabels({'0','2','4','6','8','10'})
xlabel('y (km)'); 
set(gca, 'YTick', (  0  :  100/st  : 768/st) ,'fontsize',12 ); 
yticklabels({'0','2','4','6','8','10','12','14'})
ylabel('x (km)'); 
zticks([048/st 148/st 248/st 348/st 448/st])
zticklabels({'8','6','4','2','0'})
zlabel('z (km)'); 

pbaspect([ny nx nz]);hold on;
scatter3(256/st,630/st,448/st,125,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;
scatter3(256/st,180/st,248/st,85,'*','filled','MarkerEdgeColor','k') ;

%scatter(630/st,448/st,70,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;hold on;
%scatter(180/st,248/st,185,'^','filled','MarkerEdgeColor','k','MarkerFaceColor','k') ;

%cb = colorbar; %set(cb,'position',[0.85, 0.16, 0.04, 0.165])
%title(cb,'V_{x} (m/s)');

%alpha 0.5
Vx_plot(:,128:256,:)=0;
%Vx_plot(:,250:end,:)=0;
isosurf = 1*1.2*1e-7 + 0*0.5*1e-5;
is1  = isosurface(Vx_plot, isosurf);
is2  = isosurface(Vx_plot, -isosurf);
his1 = patch(is1); set(his1,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-isosurf,'Facecolor','Flat','Edgecolor','none')

% delete *.res *.inf *.dat a.out
 