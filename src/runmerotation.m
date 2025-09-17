function [] = runme(iopt)
% various commands to compute synthetic seismograms:
% 0. compile fk.f, syn.c
% 1. run fortran fk to create greens functions.
% 2. run matlab version, fk.m
% 3. compare results
%
% usage: >> runme(iopt)
%
% 20210318 -- cralvizuri <celso.alvizuri@gmail.com>
%-----------------------------------------------------------

if nargin ~=1
    fprintf('Usage: runme(iopt)\n');
    fprintf('iopt=1  compile. run fk fortran. run fk mat.\n');
    fprintf('iopt=2  plot compare all greens functions\n');
    fprintf('iopt=3  plot compare all greens + seismograms\n');
    fprintf('iopt=4  plot compare seismograms\n');
    fprintf('iopt=10 compile+run fk. Waveforms nuclear explosion NK2017.station SEHB dist 347 km. WARNING LONG 40 hours.\n');
    fprintf('iopt=11 plot compare greens NK2017 nuclear test.\n');
    fprintf('iopt=20 runfk, run syn, compare figures. simplest example, model vsimple\n');
    %return
end

%-----------------------------------------------------------
igreen = ['0','1','2','3','4','5','6','7','8','a','b','c'];
icomp = ['z', 'r', 't'];
if (iopt==1)
    system('make clean all')                 % compile fortran fk + helper routines.
    system('fk.pl     -Mhk/5 -N512/0.1 10'); % run fortran fk. Vel model hk. Source depth 5 km. station distance 10 km. number of points 512. dt=sampling interval=0.1 sec.
    system('fk.pl -S0 -Mhk/5 -N512/0.1 10'); % explosion source  (ISO).
    fk(0);                                   % run matlab fk. input parameters are inside fk.m. fk(0) = explosions; fk(2) = double-couple. fk(1) = single force
    fk(2);
elseif (iopt==2)
    %figure(1); grid on; 
    figure('visible', 'off','Position', [0,0, 550,550]); grid on;
    for i=1:12
        fk_mat = sprintf('hk_5/10.grn.%s.mat.txt', igreen(i));
        fk_f77 = sprintf('hk_5/10.grn.%s.txt',     igreen(i));
        fprintf('loading: %s %s\n', fk_mat, fk_f77);
        mat=load(fk_mat);      % load ith greens functions
        f77=load(fk_f77);
        subplot(4,3,i); plot(f77,'b.-'); hold on; grid on; plot(mat,'r-'); legend('f77','mat'); title([fk_mat]);
    end
    print('compare_greens_hk5_d10','-dpdf')
elseif (iopt==3)
    close all;
    %figure(1); 
    figure('visible', 'off','Position', [0,0, 550,550]); grid on;
    % Compute seismograms PAS
    m0=3.3e20; mt=[1,0,0; 0,1,0;0,0,1]; dura=1; az=33.5; outnm ='outsyn'; nam='hk_5/10.grn.';
    syn(m0, mt, dura, az, outnm, nam) 
    figure('visible', 'off','Position', [0,0, 550,850]); grid on;
    for i=1:12
        fk_mat = sprintf('hk_5/10.grn.%s.mat.txt', igreen(i));
        fk_f77 = sprintf('hk_5/10.grn.%s.txt',     igreen(i));
        fprintf('loading: %s %s\n', fk_mat, fk_f77);
        mat=load(fk_mat);      % load ith greens functions
        f77=load(fk_f77);
        subplot(5,3,i); plot(f77,'b.-'); hold on; plot(mat,'r-'); grid on; legend('f77','mat'); title([fk_mat]);
    end
    for j=1:3
        isyn_c_name = sprintf('PAS.%s.txt', icomp(j));
        isyn_m_name = sprintf('PAS.%s.mat.txt', icomp(j))
        isyn_c = load(isyn_c_name);
        isyn_m = load(isyn_m_name);
        subplot(5,3,i+j); plot(isyn_c,'b.-'); hold on; plot(isyn_m,'r-'); grid on; legend('c','mat'); title(isyn_c_name);
    end
    print('compare_synseis_hk5_d10','-dpdf')
elseif (iopt==4)
    m0=3.3e20; mt=[1,0,0; 0,1,0;0,0,1]; dura=1; az=33.5; outnm ='outsyn'; nam='hk_5/10.grn.';
    syn(m0, mt, dura, az, outnm, nam) 
    close all;
    figure('visible', 'off','Position', [0,0, 550,850]); grid on;
    for j=1:3
        isyn_c_name = sprintf('PAS.%s.txt', icomp(j));
        isyn_m_name = sprintf('PAS.%s.mat.txt', icomp(j))
        isyn_c = load(isyn_c_name);
        isyn_m = load(isyn_m_name);
        subplot(3,1,j); plot(isyn_c,'b.-'); hold on; plot(isyn_m,'r-'); grid on; legend('c','mat'); title(isyn_c_name);
    end
    print('compare_synseis_hk5_d10','-dpdf')
elseif (iopt==10)
    system('make clean all')                        % compile fortran fk + helper routines.
    system('fk.pl     -MMDJ2/1 -N16384/0.05 347');  % run fortran fk. Vel model hk. Source depth 5 km. 512=number of points. 0.1=dt=sampling interval. 10=station distance, km
    system('fk.pl -S0 -MMDJ2/1 -N16384/0.05 347');  % explosion source  (ISO)
    fk(0); fk(2);                                   % run matlab fk. WARNING 40 hrs combined runtime.
elseif (iopt==11)
    %figure(1); grid on; 
    figure('visible', 'off','Position', [0,0, 550,850]); grid on;
    for i=1:12
        fk_mat = sprintf('MDJ2_1/347.grn.%s.mat.txt', igreen(i));
        fk_f77 = sprintf('MDJ2_1/347.grn.%s.txt',     igreen(i));
        fprintf('loading: %s %s\n', fk_mat, fk_f77);
        mat=load(fk_mat);      % load ith greens functions
        f77=load(fk_f77);
        subplot(4,3,i); plot(f77,'b.-'); hold on; grid on; plot(mat,'r-'); legend('f77','mat'); title([fk_mat]);
    end
    print('compare_synseis_MDJ2_d347','-dpdf')
    
    
elseif (iopt==20)   % simple model vsimple, source depth 2 km, receiver dist 1 km.
    %% COMPUTE GREENS FUNCTIONS
    tic;
    system('./fk.pl     -Mvsimple/4.0 -N4096/0.00125 9.00'); % run fortran fk. Source depth 4.0 km. station distance 2.7 km.
    % number of points 512. dt=sampling interval=0.1 sec.
    system('./fk.pl -S0 -Mvsimple/4.0 -N4096/0.00125 9.00'); % compute components explosion source (ISO)
    %2048
    %fk(0)
    %fk(2)
    %figure('visible', 'off','Position', [0,0, 550,850]); grid on;
%     for i=1:12
%         fk_mat = sprintf('vsimple_2/1.grn.%s.mat.txt', igreen(i));
%         fk_f77 = sprintf('vsimple_2/1.grn.%s.txt',     igreen(i));
%         fprintf('loading: %s %s\n', fk_mat, fk_f77);
%         mat=load(fk_mat);      % load ith greens functions
%         f77=load(fk_f77);
%         subplot(4,3,i); plot(f77,'b.-'); hold on; grid on; plot(mat,'r-'); legend('f77','mat'); title([fk_mat]);
%     end
%     print('compare_greens_vsimple2_d1','-dpdf', '-fillpage')

    %% COMPUTE SYN SEISMOGRAMS
    %m0=3.3e20; mt=[0,0,1; 0,0,0.8; 1,0.8,0]; dura=0.16; az=139; outnm ='synseis'; nam='vsimple_4.0/9.00.grn.';
    m0=3.3e20; mt=[0,0,1; 0,0,0.8; 1,0.8,0]; dura=0.16; az=139; outnm ='synseis'; nam='vsimple_4.0/9.00.grn.';dura=1;
        mt=[0.4895, 0.8403, 0.3594; 0, -0.1740, 0.0005; 0, 0, 0.2439]; 
   %mt=[0.2121    0.8071   0.2709;   0.0332    -0.5276   -0.2362;         0         0    -0.2439];

%    mt=[0.4895, 0, 0; 0, -0.1740, 0; 0, 0, 0.2439]; %az 139
%    mt=[0.2039, 0.3285, 0; 0, 0.1116, 0; 0, 0, 0.2439]; %az 0
% 
%    mt=[0, 0.8403, 0; 0.8403, 0, 0; 0, 0, 0];
%    mt=[-0.8321 ,   0.1169 ,        0;...
%         0.1169 ,   0.8321  ,       0;...
%              0  ,       0   ,      0];
% 
%    mt=[0,      0.8403, 0.3594;...
%        0.8403, 0,      0;...
%        0.3594, 0,      0     ];
%    mt=[-0.8321    0.1169   -0.2712;...
%         0.1169    0.8321   -0.2358;...
%        -0.2712   -0.2358         0];
% 
%    mt=[0.4895, 0.8403, 0.3594; ...
%        0.8403,-0.1740, 0.0005;...
%        0.3594, 0.0005, 0.2439];
%    mt=[-0.6282    0.4455   -0.2709;...
%         0.4455    0.9437   -0.2362;...
%        -0.2709   -0.2362    0.2439];

   %mt=[   -0.2121    0.3870   -0.1354;    0.3870    0.5276   -0.1181;   -0.1354   -0.1181    0.2439];
    %m0=3.3e20; mt=[-0.5,0,0;0,1,0;0,0,-0.25]; dura=0.2; az=33.5; outnm ='synseis'; nam='vsimple_1.6/0.32.grn.';
    %m0=3.3e20; mt=[1,0,0;0,1,0;0,0,1]; dura=0.2; az=0.0; outnm ='synseis'; nam='vsimple_1.6/0.32.grn.';
    outnm2=sprintf('%s.i', outnm); %az=33.5
    imt = sprintf('%d/%d/%d/%d/%d/%d', mt(1,1),mt(1,2),mt(1,3),mt(2,2),mt(2,3),mt(3,3)); % syn format: Mxx/Mxy/Mxz/Myy/Myz/Mzz
    cmd_csyn=sprintf('./syn -M%e/%s -D%d -A%f -O%s -G%s0', m0,imt,dura,az,outnm2,nam);
    fprintf('syn command cmd_csyn %s outnm %s nam %s\n', cmd_csyn, outnm, nam);
    system(cmd_csyn)                   % ACTION c
    %syn(m0, mt, dura, az, outnm, nam); % ACTION matlab
%     figure('visible', 'off','Position', [0,0, 550,850]); grid on;
%     for i=1:12
%         fk_mat = sprintf('vsimple_2/1.grn.%s.mat.txt', igreen(i));
%         fk_f77 = sprintf('vsimple_2/1.grn.%s.txt',     igreen(i));
%         fprintf('loading: %s %s\n', fk_mat, fk_f77);
%         mat=load(fk_mat);      % load ith greens functions
%         f77=load(fk_f77);
%         subplot(5,3,i); plot(f77,'b.-'); hold on; plot(mat,'r-'); grid on; legend('f77','mat'); title([fk_mat]);
%     end
    Time_Zhu_propagator = 0; 
    Time_Zhu_propagator = toc
    figure(5);% clf
    icomp = ['z', 'r', 't'];
    for j=1:3
        isyn_c_nameZ = sprintf('%s.%s.txt',     outnm, icomp(1));
        isyn_cZ = load(isyn_c_nameZ);
        %figure(6);clf;  plot(isyn_cZ,'b.-'); hold on; 
        
        isyn_c_name = sprintf('%s.%s.txt',     outnm, icomp(j));
        %isyn_m_name = sprintf('%s.%s.mat.txt', outnm, icomp(j));
        isyn_c = load(isyn_c_name);
        %isyn_m = load(isyn_m_name); isyn_m(6:end) = isyn_m(1:end-5); isyn_m(1:5) = isyn_m(1:5)*0;
        subplot(1,3,j); plot(isyn_c,'k.-'); hold on; 
        %plot(isyn_m,'r-'); 
        grid on; legend('c','mat'); title(isyn_c_name);
    end
    print('compare_synseis_vsimple2_d1','-dpdf', '-fillpage')
    % 2021-07-12 load source time function generated by syn
    %fprintf('loading source time function srctimefun.txt\n');
    srctimefun = load('srctimefun.txt');
    figure(6);clf; plot(srctimefun);

    elseif (iopt==21) % rotation.
        mt=[0,0,1; 0,0,0.8; 1,0.8,0];
        %mt=[0.4895, 0.8403, 0.3594; 0, -0.1740, 0.0005; 0, 0, 0.2439]; 
        % mt=[0.4895, 0, 0; 0, -0.1740, 0; 0, 0, 0.2439];
        % mt=[0, 0.8403, 0.3594; 0.8403, 0, 0.0005; 0.3594, 0.0005, 0]; 
        % mt=[0, 0.8403, 0; 0.8403, 0, 0; 0, 0, 0];
        % mt=[0,      0.8403, 0.3594;...
        %     0.8403, 0,      0;...
        %     0.3594, 0,      0];
         mt=[0.4895, 0.8403, 0.3594; ...
        0.8403,-0.1740, 0.0005;...
        0.3594, 0.0005, 0.2439];
        angl = 139 /180*pi;% 139 = azimuth
        %angl = 123 /180*pi;% 
        %angl = 286 /180*pi;%idist=23; iaz=286
        QQ = [cos(angl), sin(angl), 0; -sin(angl), cos(angl), 0; 0, 0, 1];
        Rot_mt = QQ*mt*(QQ') %these values go into GPU code

        rad = mt_radiat(139,mt)
else
    fprintf('stop. option not available.\n');
end
%-----------------------------------------------------------
% 



% elseif (iopt==20)   % simple model vsimple, source depth 2 km, receiver dist 1 km.
%     %% COMPUTE GREENS FUNCTIONS
%     system('./fk.pl     -Mvsimple/1.0 -N2048/0.001 2.70'); % run fortran fk. Source depth 1.0 km. station distance 2.7 km.
%     system('./fk.pl -S0 -Mvsimple/1.0 -N2048/0.001 2.70'); % compute components explosion source (ISO)
%     %% COMPUTE SYN SEISMOGRAMS
%     m0=3.3e20; mt=[0,0,1;0,0,0;1,0,0]; dura=0.2; az=0.0; outnm ='synseis'; nam='vsimple_1.0/2.70.grn.';
%     outnm2=sprintf('%s.i', outnm); %az=33.5
%     imt = sprintf('%d/%d/%d/%d/%d/%d', mt(1,1),mt(1,2),mt(1,3),mt(2,2),mt(2,3),mt(3,3)); % syn format: Mxx/Mxy/Mxz/Myy/Myz/Mzz
%     cmd_csyn=sprintf('./syn -M%e/%s -D%d -A%f -O%s -G%s0', m0,imt,dura,az,outnm2,nam);
%     fprintf('syn command cmd_csyn %s outnm %s nam %s\n', cmd_csyn, outnm, nam);
%     system(cmd_csyn)                   % ACTION c
% 
%     figure(5);% clf
%     for j=1:3
%         isyn_c_nameZ = sprintf('%s.%s.txt',     outnm, icomp(1));
%         isyn_cZ = load(isyn_c_nameZ);
%         isyn_c_name = sprintf('%s.%s.txt',     outnm, icomp(j));
%         isyn_c = load(isyn_c_name);
%         subplot(1,3,j); plot(isyn_c,'r.-'); hold on; 
% 
%         grid on; legend('c','mat'); title(isyn_c_name);
%     end
%     print('compare_synseis_vsimple2_d1','-dpdf', '-fillpage')
%     
%     srctimefun = load('srctimefun.txt');
%     figure(6);clf; plot(srctimefun);
% else