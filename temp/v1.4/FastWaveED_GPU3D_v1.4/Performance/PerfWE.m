% time for 1 iter  ; GB/s
GPUamp_MC = [0.0056 1525];
GPUamp_WE = [0.0244 791];

GPU_MC = [0.0162 529.3];
GPU_WE = [0.0704 274];

Matlab_MC = [1.044 8.22];
Matlab_WE = [9.09 2.126];

MatlabGPU_MC = [0.0386 222];
MatlabGPU_WE = [1.945 9.93];

C_MC = [10.17 0.844];
C_WE = [58.88 0.32];


%  plot memory copy
figure(1); clf;%C_MC(2)   'C++',
subplot(111); plot([Matlab_WE(2) MatlabGPU_WE(2) GPU_WE(2) GPUamp_WE(2)],'s--','MarkerSize',10,...
    'MarkerEdgeColor','black',...
    'MarkerFaceColor',[1 .6 .6] );
ylabel('MTP_{effective} (GB/s)','FontSize',11);
grid on;
xticks([1:4]);yticks([2.126   274 791]);
xticklabels({'Matlab vectorized ','Matlab vectorized gpuArray','GPU RTX 4090 Laptop','GPU Ampere A100'})
title('Elastic wave equation, N = 512^3')
text(1.9,90,'9.93')


subplot(111); plot(Matlab_WE(1)./[Matlab_WE(1) MatlabGPU_WE(1) GPU_WE(1) GPUamp_WE(1)],'s--','MarkerSize',10,...
    'MarkerEdgeColor','black',...
    'MarkerFaceColor',[1 .6 .6] );
ylabel('Acceleration x (-)','FontSize',11);
grid on;%ylim([1 200]);
yticks(fix( [Matlab_WE(1)./[Matlab_WE(1)   GPU_WE(1) GPUamp_WE(1)]] ));
xticks([1:4]);
xticklabels({'Matlab vectorized ','Matlab vectorized gpuArray','GPU RTX 4090 Laptop','GPU Ampere A100'})
title('Elastic wave equation, N = 512^3')
text(1.9,50,'4.67')
