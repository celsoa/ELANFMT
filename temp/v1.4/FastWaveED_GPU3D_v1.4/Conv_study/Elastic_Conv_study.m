tic;opengl software
clear,clf, clc;clf; format long; format compact; figure(1)
for test_n = 1:8
sc_g                = [ 0.54 0.7  1  2  3 4 5 5.5 6];
sc_g                = [ 0.7  1  2  3 4 5 5.5 6.5];
nx_g(test_n)        = 1000*sc_g(test_n) + 1*0;
nx = nx_g(test_n); sc = sc_g(test_n);

[Vx_num,Vx_dif,dx]   = YA1B_invis(nx,sc);
Vx_dif_gl(test_n)  = Vx_dif
dx_g(test_n) = dx;
end

Solution2 =   1;
figure(4); 
plot(log10(1./dx_g(:)),log10(abs(Vx_dif_gl(:)./Solution2)),'o','color',[0.8500 0.3250 0.0980],'MarkerSize',12,'linewidth',3),hold on
xlabel('$\log_{10}$(1/dx)', 'interpreter', 'latex', 'FontSize', 20)
ylabel('$\log_{10}\left|\left|err\right|\right|_1$', 'interpreter', 'latex', 'FontSize', 20)
P1  = polyfit(log10(1./dx_g(:)),log10(abs(Vx_dif_gl(:)./Solution2)),1);
L1  = log10(1./dx_g(:))*P1(1) + P1(2);
text(log10(1./dx_g(1))*1.1,log10(abs(Vx_dif_gl(1)./Solution2))*1.1,['slope = ',num2str(P1(1)),''],'interpreter', 'latex','FontSize', 24)
grid on;
plot(log10(1./dx_g(:)),L1,'-','color',[0.8500 0.3250 0.0980],'linewidth',1.5),hold on

function [Vx_num,Vx_dif,dx] = YA1B_invis(nx,sc)
%physics 
% parameters with independent units
Lx          = 1;   % size [m]
visc        = 1e-3; perm = 1e-12 /100000;
etaf_k      = visc/perm ;%/10000;  % visc / permiab  [Pa*s/m^2]
G0          = 30e9;  % shear modulus of the frame [Pa]
rho_s       = 2700;  % solid density  [kg/m^3]
K_dry       = 26e9;  % Bulk modulus of the frame [Pa]
c11         = K_dry+ 4/3*G0;
% nondimentional parameters
fi          = 0.2;           % porosity [-]
rho_fluid_rho_solid = 0.4; % ratio
K_g__K_dry  = 1.42;          % ratio
Mu_g__Mu_dry= 1.42;          % ratio
Kf_K_dry    = 0.0865;        % ratio
Tor         = 1.9;          % tortuosity [-]
% dimentionally dependent parameters
% rho_f       = rho_fluid_rho_solid*rho_s; 
K_g         = K_g__K_dry*K_dry; % solid grain material [Pa]
Mu_g        = Mu_g__Mu_dry*G0; % solid shear grain material [Pa]
K_fl        = Kf_K_dry*K_dry;   % fluid bulk modulus   [Pa]
Tor_fi      = Tor/fi;
beta_d      = 1./K_dry;            % compliance
beta_g      = 1./K_g;              % compliance
beta_f      = 1./K_fl;             % compliance
alpha       = 1 - beta_g./beta_d;  % Biot alpha 
B           = (beta_d - beta_g) ./ (beta_d - beta_g + fi.*(beta_f - beta_g)); % Biot B
rho_f       = rho_fluid_rho_solid.*rho_s; % fluid density [kg/m^3]
rho_t       = (1-fi).*rho_s + fi.*rho_f;   % total density [kg/m^3]
rho_a       = rho_f*Tor_fi;
K_u         = K_dry./(1 - B*alpha );
MM          = B.*K_u./alpha;                % biot M [Pa]

M           = [1, -alpha; -alpha, alpha/B];
iM          = inv(M); %Inv_M = [alpha/B, +alpha; +alpha, 1]/(alpha/B - alpha*alpha)
Mdvp        = [rho_t, -rho_f; -rho_f, +rho_f*Tor_fi];
iMdvp       = inv(Mdvp); 
%c11_u       = c11 + alpha^2.*MM; % c11 undrained
Pe1          = (etaf_k*Lx / (c11*rho_t)^0.5 )^-1;
%preprocessing
for counter = 1:2
rho_t       = (1-fi).*rho_s + fi.*rho_f;   % total density [kg/m^3]
nt_global    = [250*sc+1*0  850*sc+1*0];
nt           = nt_global(counter);
lamx         = 1/60*Lx;
dx          = Lx/(nx-1);
x           = (-Lx/2:dx:Lx/2)';
% initial conditions
Vx          = zeros(nx+1,1);
stress_xx          = -exp(-((x+dx*nx/4)/lamx).^2)*0;
%% 
rho_ft  = rho_f/rho_t; rho_at = rho_f*Tor_fi/rho_t; 
alphaBG = alpha/B + 4/3*G0/MM; rho12 = rho_f/rho_t;rho22 = rho_a/rho_t; Pe = 1/etaf_k(1);
Pe     = (etaf_k*Lx / (c11*rho_t)^0.5 )^-1;
c11     = 1; rho_t = 1; etaf_k = 1/Pe1;
M_EL       = [ 1 , -alpha; -alpha, (alpha/B + 4/3*G0/MM) ] ./ c11;
iM_ELan    = [  (alpha/B + 4/3*G0/MM), alpha; alpha,  1]./ (  alpha/B + 4/3*G0/MM  -alpha^2) .* c11;
Mdvp       = rho_t.*[1, -rho_ft; -rho_ft, +rho_at];
iMdvp11 = 1/rho_t.* rho_at./(rho_at - rho_ft.*rho_ft);
%% Dispersion 2
omega = 1e2;
A1_m = (alpha ^ 2 - alphaBG) * (rho12 ^ 2 - rho22) * omega ^ 4 + 1i*(alpha ^ 2 - alphaBG) * omega^3 / Pe;
A3_m = 1;
A2_m = ( (2  *rho12 * alpha - rho22 * alphaBG - 1) * omega^2) + 1i * omega * alphaBG / Pe;
Solution1 = omega./real( (( -A2_m + (A2_m.*A2_m - 4.*A3_m.*A1_m).^0.5 )./2./A3_m ).^0.5 );
Solution2 = omega./real( (( -A2_m - (A2_m.*A2_m - 4.*A3_m.*A1_m).^0.5 )./2./A3_m ).^0.5 )

Solution2 = sqrt(iM_ELan(1,1).*iMdvp11 );

dt_sound   = dx/Solution2;
rstab      = 0.45;
dt         = dt_sound*rstab;
freq = omega/2/pi; t0   = 3 / freq /dt/2.5 ; 
%%
figure(1);
for it = 1:nt  
 
    stress_xx     = stress_xx   + ( iM_ELan(1,1).*  diff(Vx,1,1)/dx ).*dt ;
    stress_xx(nx/4)  = stress_xx(nx/4)+ (1-2.* (pi .* (it - t0)*dt .*freq).^2 ) .* exp(-(pi .* (it - t0)*dt .*freq).^2) ; % source Ricker
    div_Sigmax    = diff(stress_xx,1,1)/dx;    
    Vx(2:end-1)   = (Vx(2:end-1)/dt     +  div_Sigmax.*iMdvp11 )*dt;      

end
cpus = toc;
if counter==1
[max_VxC(counter),idxC(counter)] = min(Vx(1:end));
[max_Vx(counter),idx(counter)] = min(Vx((end+1)/2-nx/4:end));

idx_lc = idx(counter)  + nx/4;
xx=idx_lc-3:1:idx_lc+3;
dydx = diff(Vx(idx_lc-3:idx_lc+3)')./diff(xx);
xc=(xx(1:end-1)+xx(2:end))/2;
ix=find(dydx(1:end-1).*dydx(2:end)<0);
xmin = interp1(dydx(ix:(ix+1)),xc(ix:(ix+1)),0);

idx(counter) = xmin - 0*nx/4;
else
    [max_VxC(counter),idxC(counter)] = min(Vx(1:end));
    [max_Vx(counter),idx(counter)] = min(Vx((end+1)/2-0*nx/4:end));   
idx_lc = idx(counter)  + (nx)/2;
xx=idx_lc-3:1:idx_lc+3;
dydx = diff(Vx(idx_lc-3:idx_lc+3)')./diff(xx);
xc=(xx(1:end-1)+xx(2:end))/2;
ix=find(dydx(1:end-1).*dydx(2:end)<0);
xmin = interp1(dydx(ix:(ix+1)),xc(ix:(ix+1)),0);
idx(counter) = xmin - 0*nx/4;
end
end
Vx_num = abs((idx(2) - idx(1))./(nt_global(2) - nt_global(1))/dt*dx) 
Vx_dif = abs(Solution2 - Vx_num);
end