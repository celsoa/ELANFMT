// This routine is written on CUDA C for a single GPU.
// 01 Aug 2024
// Copyright (C) 2024  Yury Alkhimenkov
// Massachusetts Institute of Technology

// Output: See matlab
// Input:  Parameters are generated in the Matlab script
// To use: You need a GPU with 8.0 GB of GPU DRAM; Better to use Cuda/12.0 or above
// To run and visualize: Use the Matlab script 
// 
// to compile:  see matlab
// to run:      a.exe
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GPU_ID 0
#define USE_SINGLE_PRECISION      /* Comment this line using "!" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define MPI_DAT MPI_REAL
#define PRECIS  4
#else
#define DAT     double
#define MPI_DAT MPI_DOUBLE_PRECISION
#define PRECIS  8
#endif
////////// ========== Simulation Initialisation ========== //////////
#define BLOCKS_X    32  
#define BLOCKS_Y    2  
#define BLOCKS_Z    8  
#define GRID_X      (NBX*2) 
#define GRID_Y      (NBY*32) 
#define GRID_Z      (NBZ*8) 

// maximum overlap in x, y, z direction. x : Vx is nx+1, so it is 1; y: Vy is ny+1, so it is 1; z: Vz is nz+1, so it is 1.
#define MAX_OVERLENGTH_X OVERX //3
#define MAX_OVERLENGTH_Y OVERY //3
#define MAX_OVERLENGTH_Z OVERZ //3

// Numerics
const int nx     = GRID_X*BLOCKS_X - MAX_OVERLENGTH_X;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
const int ny     = GRID_Y*BLOCKS_Y - MAX_OVERLENGTH_Y;        // we want to have some threads available for all cells of any array, also the ones that are bigger than ny.
const int nz     = GRID_Z*BLOCKS_Z - MAX_OVERLENGTH_Z;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nz.
//const int nt     =  10;
const int nt     =  3000;//1600;//1600;//3550;//2550
const int nout   =  200; 
const int niter  =  10;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Definition of basic macros
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NB_THREADS             (BLOCKS_X*BLOCKS_Y*BLOCKS_Z)
#define NB_BLOCKS              (GRID_X*GRID_Y*GRID_Z)
#define def_sizes(A,nx,ny,nz)  const int sizes_##A[] = {nx,ny,nz};                            
#define size(A,dim)            (sizes_##A[dim-1])
#define numel(A)               (size(A,1)*size(A,2)*size(A,3))
#define end(A,dim)             (size(A,dim)-1)
#define zeros_h(A,nx,ny,nz)    def_sizes(A,nx,ny,nz);                                  \
                               DAT *A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }
#define zeros(A,nx,ny,nz)      def_sizes(A,nx,ny,nz);                                         \
                               DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }        \
                               cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                               cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define ones(A,nx,ny,nz)       def_sizes(A,nx,ny,nz);                                         \
                               DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)1.0; }        \
                               cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                               cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define gather(A)              cudaMemcpy( A##_h,A##_d,numel(A)*sizeof(DAT),cudaMemcpyDeviceToHost);
#define free_all(A)            free(A##_h);cudaFree(A##_d);
#define swap(A,B,tmp)          DAT *tmp; tmp = A##_d; A##_d = B##_d; B##_d = tmp;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables for cuda
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int nprocs=1, me=0;
dim3 grid, block;
int gpu_id=-1;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions (host code)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_up_gpu(){
    block.x = BLOCKS_X; block.y = BLOCKS_Y; block.z = BLOCKS_Z;
    grid.x  = GRID_X;   grid.y  = GRID_Y;   grid.z  = GRID_Z;
    gpu_id  = GPU_ID;
    cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset();                                // Reset the device to avoid problems caused by a crash in a previous run (does still not assure proper working in any case after a crash!).
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
}

void clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}
//// Timer 
// Timer 
int saveData = 0;
DAT GPUinfo[3];
cudaEvent_t startD, stopD;
float milliseconds = 0;
//#include "sys/time.h"
//double timer_start = 0;
//double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
//void   tic(){ timer_start = cpu_sec(); }
//double toc(){ return cpu_sec()-timer_start; }
//void   tim(const char *what, double n){ double s=toc();if(me==0){ printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); } }
//////////// ========== Save & Read Data functions ========== //////////
/// Params to be saved for evol plot ///
#include <stdarg.h> /* needed for va_*         */
int vscprintf(const char* format, va_list ap)
{
    va_list ap_copy;
    va_copy(ap_copy, ap);
    int retval = vsnprintf(NULL, 0, format, ap_copy);
    va_end(ap_copy);
    return retval;
}
int vasprintf(char** strp, const char* format, va_list ap)
{
    int len = vscprintf(format, ap);
    if (len == -1)
        return -1;
    char* str = (char*)malloc((size_t)len + 1);
    if (!str)
        return -1;
    int retval = vsnprintf(str, len + 1, format, ap);
    if (retval == -1) {
        free(str);
        return -1;
    }
    *strp = str;
    return retval;
}
int asprintf(char** strp, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    int retval = vasprintf(strp, format, ap);
    va_end(ap);
    return retval;
}
#define NB_PARAMS  1
void save_info(){
    FILE* fid;
    if (me==0){ fid=fopen("0_infos.inf", "w"); fprintf(fid,"%d %d %d %d %d %d",PRECIS,nx,ny,nz,NB_PARAMS,(int)ceil((DAT)niter/(DAT)nout));  fclose(fid);}
}

void save_array(DAT* A, size_t nb_elems, const char A_name[], int isave){
    char* fname; FILE* fid;
    asprintf(&fname, "%d_%d_%s.res" ,isave, me, A_name); 
    fid=fopen(fname, "wb"); fwrite(A, PRECIS, nb_elems, fid); fclose(fid); free(fname);
}
#define SaveArray(A,A_name)  gather(A); save_array(A##_h, numel(A), A_name, isave);

void read_data(DAT* A_h, DAT* A_d, int nx,int ny,int nz, const char A_name[], const char B_name[],int isave){
    char* bname; size_t nb_elems = nx*ny*nz; FILE* fid;
    asprintf(&bname, "%d_%d_%s.%s", isave, me, A_name, B_name);
    fid=fopen(bname, "rb"); // Open file
    if (!fid){ fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
    fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
    cudaMemcpy(A_d, A_h, nb_elems*sizeof(DAT), cudaMemcpyHostToDevice);
    if (me==0) printf("Read data: %d files %s.%s loaded (size = %dx%dx%d) \n", nprocs,A_name,B_name,nx,ny,nz); free(bname);
}

void read_data_h(DAT* A_h, int nx, int ny,int nz, const char A_name[], const char B_name[],int isave){
    char* bname; size_t nb_elems = nx*ny*nz; FILE* fid;
    asprintf(&bname, "%d_%d_%s.%s", isave, me, A_name, B_name);
    fid=fopen(bname, "rb"); // Open file
    if (!fid){ fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
    fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
    if (me==0) printf("Read data: %d files %s.%s loaded (size = %dx%dx%d) \n", nprocs,A_name,B_name,nx,ny,nz); free(bname);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define Dx  ( (DAT)1.0/dx )
#define Dy  ( (DAT)1.0/dy )
#define Dz  ( (DAT)1.0/dz )

#define load(A,nx,ny,Aname) DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*sizeof(DAT));  \
                            FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(DAT), (nx)*(ny), A##fid); fclose(A##fid); \
                            cudaMalloc(&A##_d,((nx)*(ny))*sizeof(DAT)); \
                            cudaMemcpy(A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);  
#define  swap(A,B,tmp)      DAT *tmp; tmp = A##_d; A##_d = B##_d; B##_d = tmp;

#define load3(A,nx,ny,nz,Aname)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*(nz)*sizeof(DAT));  \
                             FILE* A##fid=fopen(Aname, "rb"); fread(A##_h, sizeof(DAT), (nx)*(ny)*(nz), A##fid); fclose(A##fid); \
                             cudaMalloc(&A##_d,((nx)*(ny)*(nz))*sizeof(DAT)); \
                             cudaMemcpy(A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice); 
#define   push(A,nx,ny,nz) cudaMemcpy( A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice);

// Source
const DAT A0    = 1e9;   // amplitude of the perturbation 1e3
const DAT f0    = 2e3;    // peak freq.
const int ds    = 2.0;
#define x_s    (Lx - Lx/2.0)
#define y_s    (Ly - Ly/2.0)
#define z_s    (Lz - Lz/2.0)
#define pi (DAT)3.14159265358979323846

void pert(const int nt, const DAT f0, const DAT A0, DAT dt, DAT* Src){
    int itr;
    DAT t0 = (DAT)30.0/3e4/dt;///dt
    for (itr=0; itr<nt; itr++){
        //DAT t    = ((DAT)itr)*dt;
        //Src[itr] = A0*exp(-(DAT)0.5*((DAT)4.0*f0)*((DAT)4.0*f0)*(t-t0)*(t-t0));
        Src[itr] = A0 * (  (DAT)1.0 - (DAT)2.0*(pi * (itr - t0)*dt *f0)*(pi * (itr - t0)*dt *f0)  ) * exp( -(pi * (itr - t0)*dt*f0) * (pi * (itr - t0)*dt*f0) );
    }
}

// Computing physics kernels /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void init(DAT* x, DAT* y,DAT* z, DAT* Prf, DAT* sigma_xx, DAT* sigma_yy, DAT* sigma_zz, DAT* Vx, DAT* Vy, DAT* Qxft, DAT* Qyft, const DAT dx, const DAT dy,const DAT dz, const DAT Lx, const DAT Ly,const DAT Lz, const DAT lamx, const DAT lamy,const DAT lamz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    if (iz<nz && iy<ny && ix<nx){ x[ix + iy*nx + iz*nx*ny] = (DAT)ix*dx - (DAT)0.5*Lx; }
    if (iz<nz && iy<ny && ix<nx){ y[ix + iy*nx + iz*nx*ny] = (DAT)iy*dy - (DAT)0.5*Ly; }
    if (iz<nz && iy<ny && ix<nx){ z[ix + iy*nx + iz*nx*ny] = (DAT)iz*dz - (DAT)0.5*Lz; }
    //if (iz<nz && iy<ny && ix<nx){ sigma_xx[ix + iy*nx + iz*nx*ny] = -(DAT)10000000000.0*exp(  -(x[ix + iy*nx + iz*nx*ny]*x[ix + iy*nx + iz*nx*ny]/lamx/lamx) -(y[ix + iy*nx + iz*nx*ny]*y[ix + iy*nx + iz*nx*ny]/lamy/lamy) -(z[ix + iy*nx + iz*nx*ny]*z[ix + iy*nx + iz*nx*ny]/lamz/lamz)  ); }
    //if (iz<nz && iy<ny && ix<nx){ sigma_yy[ix + iy*nx + iz*nx*ny] = -(DAT)10000000000.0*exp(  -(x[ix + iy*nx + iz*nx*ny]*x[ix + iy*nx + iz*nx*ny]/lamx/lamx) -(y[ix + iy*nx + iz*nx*ny]*y[ix + iy*nx + iz*nx*ny]/lamy/lamy) -(z[ix + iy*nx + iz*nx*ny]*z[ix + iy*nx + iz*nx*ny]/lamz/lamz)  ); }
    //if (iz<nz && iy<ny && ix<nx){ sigma_zz[ix + iy*nx + iz*nx*ny] = -(DAT)10000000000.0*exp(  -(x[ix + iy*nx + iz*nx*ny]*x[ix + iy*nx + iz*nx*ny]/lamx/lamx) -(y[ix + iy*nx + iz*nx*ny]*y[ix + iy*nx + iz*nx*ny]/lamy/lamy) -(z[ix + iy*nx + iz*nx*ny]*z[ix + iy*nx + iz*nx*ny]/lamz/lamz)  ); }
}

__global__ void source(DAT* x, DAT* y,DAT* z, DAT* Prf, DAT* sigma_xx, DAT* sigma_yy, DAT* sigma_zz,DAT* sigma_xy,DAT* sigma_xz,DAT* sigma_yz, DAT* Vx, DAT* Vy, DAT* Qxft, DAT* Qyft,DAT* Src, const DAT dx, const DAT dy,const DAT dz, const DAT Lx, const DAT Ly,const DAT Lz, const DAT lamx, const DAT lamy,const DAT lamz, const int nx, const int ny, const int nz, int it, int xsrc, int ysrc, int zsrc, int ds, DAT dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    // Explosion m00, m01, m02, m11, m12, m22 M0 = 7.413e+23     >[0.6980 - 0.0609  0.0165  0.5766  0.3500  0.9630]
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds) && iy>(ysrc) && ix<(xsrc+ds) && ix>(xsrc)){ sigma_xx[ix + iy*nx + iz*nx*ny]  = sigma_xx[ix + iy*nx + iz*nx*ny]+ (DAT)0.698*Src[it]  ; }
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds) && iy>(ysrc) && ix<(xsrc+ds) && ix>(xsrc)){ sigma_yy[ix + iy*nx + iz*nx*ny]  = sigma_yy[ix + iy*nx + iz*nx*ny]+ (DAT)0.5766*Src[it]  ; }
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds) && iy>(ysrc) && ix<(xsrc+ds) && ix>(xsrc)){ sigma_zz[ix + iy*nx + iz*nx*ny]  = sigma_zz[ix + iy*nx + iz*nx*ny]+ (DAT)0.963*Src[it]  ; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 1) && iy>(ysrc + 1) && ix<(xsrc + ds + 1) && ix>(xsrc + 1)) { sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] = sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] + (DAT)0.3500 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 2) && iy>(ysrc + 2) && ix<(xsrc + ds + 2) && ix>(xsrc + 2)) { sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)0.0165 * Src[it]; }
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)]  = sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)] - (DAT)0.0609*Src[it]  ; }
    
    // Collaps m00, m01, m02, m11, m12, m22 M0 = 2.512e+22     >[-0.8262 - 0.1622 - 0.0976 - 0.8150 - 0.1403 - 0.7363]
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds) && iy>(ysrc) && ix<(xsrc + ds) && ix>(xsrc)) { sigma_xx[ix + iy * nx + iz * nx * ny] = sigma_xx[ix + iy * nx + iz * nx * ny] - (DAT)0.8262 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds) && iy>(ysrc) && ix<(xsrc + ds) && ix>(xsrc)) { sigma_yy[ix + iy * nx + iz * nx * ny] = sigma_yy[ix + iy * nx + iz * nx * ny] - (DAT)0.815 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds) && iy>(ysrc) && ix<(xsrc + ds) && ix>(xsrc)) { sigma_zz[ix + iy * nx + iz * nx * ny] = sigma_zz[ix + iy * nx + iz * nx * ny] - (DAT)0.7363 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 1) && iy>(ysrc + 1) && ix<(xsrc + ds + 1) && ix>(xsrc + 1)) { sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] = sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] - (DAT)0.1403 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 2) && iy>(ysrc + 2) && ix<(xsrc + ds + 2) && ix>(xsrc + 2)) { sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] - (DAT)0.0976 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 2) && iy>(ysrc + 2) && ix<(xsrc + ds + 2) && ix>(xsrc + 2)) { sigma_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] = sigma_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] - (DAT)0.1622 * Src[it]; }

	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)]  = sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)] + (DAT)0.8*Src[it]  ; }
	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+1) && iy>(ysrc+1) && ix<(xsrc+ds+1) && ix>(xsrc+1)){ sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)]  = sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)] + (DAT)0.8*Src[it]  ; }
    if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 1) && iy>(ysrc + 1) && ix<(xsrc + ds + 2) && ix>(xsrc + 2)) { sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] = sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] + (DAT)0.8 * Src[it]; }
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+1) && iy>(ysrc+1) && ix<(xsrc+ds+1) && ix>(xsrc+1)){ sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)]  = sigma_yz[ix + iy* nx    + (iz+1)* nx   *(ny+1)] + (DAT)0.4*Src[it]  ; }

    if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 0) && iy>(ysrc + 0) && ix<(xsrc + ds - 2) && ix>(xsrc - 2)) { sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)1.0 * Src[it]; }
    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xz[ix + iy*(nx+1) + (iz+1)*(nx+1)* ny   ]  = sigma_xz[ix + iy*(nx+1) + (iz+1)*(nx+1)* ny   ] + (DAT)0.25*Src[it]  ; }

	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ]  = sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ] + (DAT)1.0*Src[it]  ; }

	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+1) && iy>(ysrc+1) && ix<(xsrc+ds+1) && ix>(xsrc+1)){ sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)]  = sigma_yz[ix + iy* nx    + (iz+1)* nx   *(ny+1)] + (DAT)0.4*Src[it]  ; }

	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xz[ix + iy*(nx+1) + (iz+1)*(nx+1)* ny   ]  = sigma_xz[ix + iy*(nx+1) + (iz+1)*(nx+1)* ny   ] + (DAT)0.25*Src[it]  ; }

	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ]  = sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ] + (DAT)0.25*Src[it]  ; }
	//if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds+2) && iy>(ysrc+2) && ix<(xsrc+ds+2) && ix>(xsrc+2)){ sigma_xz[ix+1 + iy*(nx+1) + iz*(nx+1)* ny   ]  = sigma_xz[ix+1 + iy*(nx+1) + iz*(nx+1)* ny   ] + (DAT)0.25*Src[it]  ; }
    
    // Synt EX
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 1) && iy>(ysrc + 1) && ix<(xsrc + ds + 1) && ix>(xsrc + 1)) { sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] = sigma_yz[ix + iy * nx + iz * nx * (ny + 1)] + (DAT)0.8 * Src[it]; }
    //if (iz<(zsrc + ds) && iz>(zsrc) && iy<(ysrc + ds + 2) && iy>(ysrc + 2) && ix<(xsrc + ds + 2) && ix>(xsrc + 2)) { sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] = sigma_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] + (DAT)1.0 * Src[it]; }

}

__global__ void record(DAT* x, DAT* y,DAT* z, DAT* Prf, DAT* sigma_xx, DAT* sigma_yy, DAT* sigma_zz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Rec,DAT* Rec_x,DAT* Rec_y,DAT* Rec_y2, const DAT dx, const DAT dy,const DAT dz, const DAT Lx, const DAT Ly,const DAT Lz, const DAT lamx, const DAT lamy,const DAT lamz, const int nx, const int ny, const int nz, int it, int xsrc, int ysrc, int zsrc, int ds, DAT dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds) && iy>(ysrc) && ix<(xsrc+ds) && ix>(xsrc))
	if (iz>0 && iz<2 && iy>128 && iy<130 && ix>503 && ix<505){ Rec[it]  = (DAT)0.5*Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )] + (DAT)0.5*Vz[ix+0+(iy+0)*(nx) +iz*(nx  )*(ny )]  ;} 
	if (iz>0 && iz<2 && iy>128 && iy<130 && ix>503 && ix<505){ Rec_x[it]  = (DAT)0.5*Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] + (DAT)0.5*Vx[ix+(iy+0)*(nx+1)+ iz*(nx+1)*(ny  )]; }
	if (iz>0 && iz<2 && iy>131 && iy<133 && ix>503 && ix<505){ Rec_y[it]  = Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   ; }
	if (iz>0 && iz<2 && iy>130 && iy<132 && ix>503 && ix<505){ Rec_y2[it] = Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   ; }
}

__global__ void BC1(DAT* x, DAT* y,DAT* z, DAT* Prf, DAT* sigma_xx, DAT* sigma_yy, DAT* sigma_zz, DAT* sigma_xy, DAT* sigma_xz, DAT* sigma_yz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Rec, const DAT dx, const DAT dy,const DAT dz, const DAT Lx, const DAT Ly,const DAT Lz, const DAT lamx, const DAT lamy,const DAT lamz, const int nx, const int ny, const int nz, int it, int xsrc, int ysrc, int zsrc, int ds, DAT dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    //if (iz<(zsrc+ds) && iz>(zsrc) && iy<(ysrc+ds) && iy>(ysrc) && ix<(xsrc+ds) && ix>(xsrc))
	//if (iz==0 && iy<nx && ix<nx){ sigma_xx[ix + iy*nx + iz*nx*ny]  = (DAT)0.0   ; } 
	//if (iz==0 && iy<nx && ix<nx){ sigma_yy[ix + iy*nx + iz*nx*ny]  = (DAT)0.0   ; } 
	if (iz==0 && iy<nx && ix<nx){ sigma_zz[ix + iy*nx + iz*nx*ny]  = (DAT)0.0   ; } 
	//if (iz==0 && iy>0 && iy<ny && ix>0 && ix<nx){ sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)]  = (DAT)0.0   ; } 
	if (iz==1 && iy<nx && ix>0 && ix<nx){ 		  sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ]  = ((DAT)1.0/(DAT)3.0)*sigma_xz[ix + iy*(nx+1) + (iz+1)*(nx+1)* ny   ]   ; } 
	if (iz==1 && iy>0 && iy<ny && ix<nx){ 		  sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)]  = ((DAT)1.0/(DAT)3.0)*sigma_yz[ix + iy* nx    + (iz+1)* nx   *(ny+1)]   ; } 
}

__global__ void compute_StressPrf(DAT* Prf, DAT* sigma_xx, DAT* sigma_yy,DAT* sigma_zz, DAT* sigma_xy, DAT* sigma_xz, DAT* sigma_yz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Qzft, DAT* c11u_het, DAT* c12u_het,DAT* weights_x, DAT* weights_y, DAT* weights_z, DAT* weights_xy, DAT* weights_xz, DAT* weights_yz, const DAT dx, const DAT dy,const DAT dz,const DAT dt,const DAT c11u,const DAT c22u,const DAT c33u,const DAT c12u,const DAT c13u,const DAT c23u,const DAT c44u,const DAT c55u,const DAT c66u,const DAT alpha1,const DAT alpha2,const DAT alpha3,const DAT M1, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    #define diffVx  (   (Vx[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]   -  Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   )*((DAT)1.0/dx)   )
    #define diffVy  (   (Vy[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]   -  Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   )*((DAT)1.0/dy)   )
    #define diffVz  (   (Vz[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]   -  Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   )*((DAT)1.0/dz)   )
    //#define div_Qf  ( (((DAT)1.0/dx)*(Qxft[(ix+1) + (iy  )*(nx+1) + (iz  )*(nx+1)*(ny  )]-Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )])  + ((DAT)1.0/dy)*(Qyft[(ix  ) + (iy+1)*(nx  ) + (iz  )*(nx  )*(ny+1)]-Qyft[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]) + ((DAT)1.0/dz)*(Qzft[(ix  ) + (iy  )*(nx  ) + (iz+1)*(nx  )*(ny  )]-Qzft[ix + iy*(nx  ) + iz*(nx  )*(ny  )]) )    )

    if (iz<nz && iy<ny && ix<nx){
        DAT Vxx = diffVx;
        DAT Vyy = diffVy;
        DAT Vzz = diffVz;
        //DAT Qff = div_Qf; weights_x[ix + iy * nx + iz * nx * ny] *  weights_y[ix + iy * nx + iz * nx * ny] * weights_z[ix + iy * nx + iz * nx * ny] *
        sigma_xx[ix + iy*nx + iz*nx*ny]     = weights_x[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * (sigma_xx[ix + iy*nx + iz*nx*ny]  + dt*(c11u_het[ix + iy*nx + iz*nx*ny]* Vxx + c12u_het[ix + iy*nx + iz*nx*ny]* Vyy + c12u_het[ix + iy*nx + iz*nx*ny]* Vzz )); //+ alpha1*M1*Qff
        sigma_yy[ix + iy*nx + iz*nx*ny]     = weights_y[ix + iy * (nx)+iz * (nx) * (ny + 1)] * (sigma_yy[ix + iy*nx + iz*nx*ny]  + dt*(c12u_het[ix + iy*nx + iz*nx*ny]* Vxx + c11u_het[ix + iy*nx + iz*nx*ny]* Vyy + c12u_het[ix + iy*nx + iz*nx*ny]* Vzz )); //+ alpha2*M1*Qff
        sigma_zz[ix + iy*nx + iz*nx*ny]     = weights_z[ix + iy * (nx)+iz * (nx) * (ny)] * (sigma_zz[ix + iy*nx + iz*nx*ny]  + dt*(c12u_het[ix + iy*nx + iz*nx*ny]* Vxx + c12u_het[ix + iy*nx + iz*nx*ny]* Vyy + c11u_het[ix + iy*nx + iz*nx*ny]* Vzz )); //+ alpha3*M1*Qff
        //Prf[ix + iy*nx + iz*nx*ny] = Prf[ix + iy*nx + iz*nx*ny] + dt*(  -alpha1* M1* Vxx  - alpha2* M1* Vyy - alpha3* M1* Vzz - M1* Qff );
    }
    if (iz<nz && iy>0 && iy<ny && ix>0 && ix<nx){ sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)] = weights_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] * (sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)]  + c66u*dt*( (Vy[ix + iy* nx + iz*nx  *(ny+1)] - Vy[ix-1 + iy* nx + iz*nx*(ny+1)  ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)*(ny-0)] - Vx[ix  +(iy-1)*(nx+1) +  iz   *(nx+1)*ny])*((DAT)1.0/dy) )); }
    if (iz>0 && iz<nz && iy<ny && ix>0 && ix<nx){ sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ] = weights_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] * (sigma_xz[ix + iy*(nx+1) + iz*(nx+1)* ny   ]  + c55u*dt*( (Vz[ix + iy* nx + iz*nx  * ny   ] - Vz[ix-1 + iy* nx + iz*nx* ny     ])*((DAT)1.0/dx) + (Vx[ix + iy* (nx+1)+ iz*(nx+1)* ny   ] - Vx[ix + iy* (nx+1)    + (iz-1)*(nx+1)*ny])*((DAT)1.0/dz) )); }
    if (iz>0 && iz<nz && iy>0 && iy<ny && ix<nx){ sigma_yz[ix + iy* nx    + iz* nx   *(ny+1)] = weights_yz[ix + iy * nx + iz * nx * (ny + 1)] * (sigma_yz[ix + iy* nx    + iz*nx*(ny+1)    ]  + c44u*dt*( (Vy[ix + iy* nx + iz*nx*  (ny+1)] - Vy[ix + iy* nx + (iz-1)*nx*(ny+1)])*((DAT)1.0/dz) + (Vz[ix + iy* nx    + iz*nx*ny        ] - Vz[ix + (iy-1)* nx    +  iz   * nx   *ny])*((DAT)1.0/dy) )); }
    //weights_xy[ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)] *  weights_xz[ix + iy * (nx + 1) + iz * (nx + 1) * ny] *   weights_yz[ix + iy * nx + iz * nx * (ny + 1)] *
    #undef diffVx
    #undef diffVy
    #undef diffVz
    //#undef div_Qf
}

__global__ void update_Qxft(DAT* Prf, DAT* sigma_xx, DAT* sigma_yy,DAT* sigma_zz, DAT* sigma_xy, DAT* sigma_xz, DAT* sigma_yz, DAT* Vx, DAT* Vy,DAT* Vz, DAT* Qxft, DAT* Qyft,DAT* Qzft,DAT* Qxold,DAT* Qyold,DAT* Qzold,DAT* vrhox11het,DAT* weights_x, DAT* weights_y, DAT* weights_z, const DAT dx, const DAT dy,const DAT dz,const DAT dt,const DAT eta_k1,const DAT eta_k2,const DAT eta_k3,const DAT vrho_x_11, const DAT vrho_y_11,const DAT vrho_z_11,const DAT vrho_x_12,const DAT vrho_y_12,const DAT vrho_z_12,const DAT vrho_x_22,const DAT vrho_y_22,const DAT vrho_z_22, const int nx, const int ny, const int nz, const DAT chi){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

    #define div_Sigmax  ( ( sigma_xx[ix + iy*nx + iz*nx*ny]-sigma_xx[(ix-1) + (iy  )*nx + (iz  )*nx*ny] )*((DAT)1.0/dx) + ( sigma_xy[ix   + (iy+1)*(nx+1) + iz*(nx+1)*(ny+1)] - sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dy) + (sigma_xz[ix + (iy+0)*(nx+1) + (iz+1)* (nx+1)*(ny+0)] - sigma_xz[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dz) )
    #define div_Sigmay  ( ( sigma_yy[ix + iy*nx + iz*nx*ny]-sigma_yy[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) + ( sigma_xy[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+1)] - sigma_xy[ix + iy*(nx+1) + iz*(nx+1)*(ny+1)])*((DAT)1.0/dx) + (sigma_yz[ix + (iy  )*(nx+0) + (iz+1)* (nx+0)*(ny+1)] - sigma_yz[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dz) )
    #define div_Sigmaz  ( ( sigma_zz[ix + iy*nx + iz*nx*ny]-sigma_zz[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ( sigma_xz[ix+1 + (iy  )*(nx+1) + iz*(nx+1)*(ny+0)] - sigma_xz[ix + iy*(nx+1) + iz*(nx+1)*(ny+0)])*((DAT)1.0/dx) + (sigma_yz[ix + (iy+1)*(nx+0) +  iz   * (nx+0)*(ny+1)] - sigma_yz[ix + iy*(nx+0) + iz*(nx+0)*(ny+1)])*((DAT)1.0/dy) )
    //#define Q_gradPrfx  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix-1) + (iy  )*nx + (iz  )*nx*ny] )*((DAT)1.0/dx) + ((DAT)1.0-chi )*( Qxold[ix + iy*(nx+1) +  iz*  (nx+1)*ny   ] )*eta_k1)
    //#define Q_gradPrfy  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix  ) + (iy-1)*nx + (iz  )*nx*ny] )*((DAT)1.0/dy) + ((DAT)1.0-chi )*( Qyold[ix + iy*(nx  ) + (iz  )*nx*  (ny+1)] )*eta_k2)
    //#define Q_gradPrfz  ( ( Prf[ix + iy*nx + iz*nx*ny]  -Prf[(ix  ) + (iy  )*nx + (iz-1)*nx*ny] )*((DAT)1.0/dz) + ((DAT)1.0-chi )*( Qzold[ix + iy*(nx  ) +  iz*   nx*   ny   ] )*eta_k3)
    
    //weights_x[ix + iy * nx + iz * nx * ny]* weights_y[ix + iy * nx + iz * nx * ny]* weights_z[ix + iy * nx + iz * nx * ny]*

    if (iz>0 && iz<(nz-1) && iy>0 && iy<(ny-1) && ix>0 && ix<nx){//vrho_x_11 (nx+1)
        //DAT QPx = Q_gradPrfx;
        DAT dSx = div_Sigmax;
        //Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] = ( Qxold[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]*((DAT)1.0/dt) - vrho_x_22* QPx - vrho_x_12* dSx )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*vrho_x_22*eta_k1 ));
        Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]   = weights_x[ix + iy * (nx + 1) + iz * (nx + 1) * (ny)] * ( Vx[ix + iy*(nx+1) + iz*(nx+1)*(ny  )]*((DAT)1.0/dt) + vrho_x_11 * dSx ) * dt;
    }   //+ vrho_x_12* ( QPx + chi*eta_k1*Qxft[ix + iy*(nx+1) + iz*(nx+1)*(ny  )] )     vrhox11het[(ix-1) + (iy  )*(nx-1) + (iz  )*(nx-1)*ny]
    if (iz>0 && iz<(nz-1) && iy>0 && iy<ny && ix>0 && ix<(nx-1)){
        //DAT QPy = Q_gradPrfy;
        DAT dSy = div_Sigmay;
        //Qyft[ix + iy*(nx  ) + iz*(nx  )*(ny+1)] = ( Qyold[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]*((DAT)1.0/dt) - vrho_y_22* QPy - vrho_y_12* dSy )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*vrho_y_22*eta_k2 )  );
        Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]   = weights_y[ix + iy * (nx)+iz * (nx) * (ny + 1)] * ( Vy[ix + iy*(nx  ) + iz*(nx  )*(ny+1)]*((DAT)1.0/dt) + vrho_y_11* dSy ) * dt;
    } //+ vrho_y_12* ( QPy + chi*eta_k2*Qyft[ix + iy*(nx  )   + iz*(nx  )*(ny+1)] )
    if (iz>0 && iz<nz && iy>0 && iy<(ny-1) && ix>0 && ix<(nx-1)){
        //DAT QPz = Q_gradPrfz;
        DAT dSz = div_Sigmaz;
        //Qzft[ix + iy*(nx  ) + iz*(nx  )*(ny  )] = ( Qzold[ix + iy*(nx  ) + iz*(nx  )*(ny  )]*((DAT)1.0/dt) - vrho_z_22* QPz - vrho_z_12* dSz )* (  (DAT)1.0/( ((DAT)1.0/dt) + chi*vrho_z_22*eta_k3 )  );
        Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]   = weights_z[ix + iy * (nx)+iz * (nx) * (ny)] * ( Vz[ix + iy*(nx  ) + iz*(nx  )*(ny  )]*((DAT)1.0/dt) + vrho_z_11* dSz ) * dt;
    } //+ vrho_z_12* ( QPz + chi*eta_k3*Qzft[ix + iy*(nx  )   + iz*(nx  )*(ny  )] )

    #undef div_Sigmax
    #undef div_Sigmay
    #undef div_Sigmaz
    //#undef Q_gradPrfx
    //#undef Q_gradPrfy
    //#undef Q_gradPrfz
    //#undef Q_gradPrfx1
    //#undef Q_gradPrfy1
    //#undef Q_gradPrfz1
}
////////// ========================================  MAIN  ======================================== //////////
int main(){
    size_t i, N;
    int it;
    N = nx*ny*nz; DAT mem = (DAT)1e-9*(DAT)N*sizeof(DAT);
    set_up_gpu();
    printf("\n  -------------------------------------------------------------------------- ");
    printf("\n  | FastBiot_GPU3D_v1:  Wave propagation in anisotropic poroelastic media  | ");
    printf("\n  --------------------------------------------------------------------------  \n\n");
    printf("Local size: %dx%dx%d (%1.4f GB) %d iterations ...\n", nx, ny, nz, mem*19.0, nt);
    printf("Launching (%dx%dx%d) grid of (%dx%dx%d) blocks.\n\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    // Load input parameters
    load(pa1,NPARS1 , 1,"pa1.dat")
    load(pa2,NPARS2 , 1,"pa2.dat")
    load(pa3,NPARS3 , 1,"pa3.dat") //dt   =pa1_h[3]
    DAT dx  =pa1_h[0],dy  =pa1_h[1],dz  =pa1_h[2],dt  =pa1_h[3],Lx  =pa1_h[4],Ly  =pa1_h[5],Lz  =pa1_h[6],lamx=pa1_h[7],lamy  =pa1_h[8],lamz   =pa1_h[9];
    DAT c11u=pa2_h[0],c33u=pa2_h[1],c13u=pa2_h[2],c12u=pa2_h[3],c23u=pa2_h[4],c44u=pa2_h[5],c55u=pa2_h[6],c66u=pa2_h[7],alpha1=pa2_h[8],alpha2 =pa2_h[9],alpha3=pa2_h[10],M1=pa2_h[11],c22u=pa2_h[12];
    //double eta_k1=pa3_h[0],eta_k2=pa3_h[1],eta_k3=pa3_h[2],mm1=pa3_h[3],mm2=pa3_h[4],mm3=pa3_h[5],delta1=pa3_h[6],delta2 =pa3_h[7],delta3=pa3_h[8],rho_fluid1=pa3_h[9],rho_fluid2=pa3_h[10],rho_fluid3 =pa3_h[11],rho1=pa3_h[12],rho2=pa3_h[13],rho3=pa3_h[14];
    DAT vrho_x_11=pa3_h[0],vrho_y_11=pa3_h[1],vrho_z_11=pa3_h[2],vrho_x_12=pa3_h[3],vrho_y_12=pa3_h[4],vrho_z_12=pa3_h[5],vrho_x_22=pa3_h[6],vrho_y_22=pa3_h[7],vrho_z_22=pa3_h[8], eta_k1=pa3_h[9],eta_k2=pa3_h[10],eta_k3=pa3_h[11];
    const DAT chi    = (DAT)0.5;

//load3(vrhox11het,nx-1 , ny,nz,"vrho_x_11_hc.dat");
//def_sizes(vrhox11het,nx-1,ny,nz); 
load3(c11u_het,nx , ny,nz,"c11u_het.dat");
def_sizes(c11u_het,nx,ny,nz); 
load3(c12u_het,nx , ny,nz,"c12u_het.dat");
def_sizes(c12u_het,nx,ny,nz); 

load3(weights_x,nx+1 , ny,nz,"weights_x.dat");
def_sizes(weights_x,nx,ny,nz);
load3(weights_y, nx, ny+1, nz, "weights_y.dat");
def_sizes(weights_y, nx, ny, nz);
load3(weights_z, nx, ny, nz+1, "weights_z.dat");
def_sizes(weights_z, nx, ny, nz);

load3(weights_xy, nx, ny, nz, "weights_xy.dat");
def_sizes(weights_xy, nx+1, ny+1, nz);
load3(weights_xz, nx, ny, nz, "weights_xz.dat");
def_sizes(weights_xz, nx + 1, ny , nz + 1);
load3(weights_yz, nx, ny, nz, "weights_yz.dat");
def_sizes(weights_yz, nx , ny + 1, nz + 1);

    // Initial arrays
    zeros(x        ,8  ,8  ,8  );
    zeros(y        ,8  ,8  ,8  );
    zeros(z        ,8  ,8  ,8  );
    zeros(Vx       ,nx+1,ny  ,nz  );
    zeros(Vy       ,nx  ,ny+1,nz  );
    zeros(Vz       ,nx  ,ny  ,nz+1);
    zeros(sigma_xx ,nx  ,ny  ,nz  );
    zeros(sigma_yy ,nx  ,ny  ,nz  );
    zeros(sigma_zz ,nx  ,ny  ,nz  );
    zeros(sigma_xy ,nx+1,ny+1,nz  );
    zeros(sigma_xz ,nx+1,ny  ,nz+1);
    zeros(sigma_yz ,nx  ,ny+1,nz+1);
    zeros(Prf      ,8  ,8  ,8  );
    zeros(Qxft     ,8+1,8  ,8  );
    zeros(Qyft     ,8  ,8+1,8  );
    zeros(Qzft     ,8  ,8,8+1  );
    zeros(Qxold    ,8+1,8  ,8  );
    zeros(Qyold    ,8  ,8+1,8  );
    zeros(Qzold    ,8  ,8,8+1  );

    zeros(vrhox11het        ,8  ,8  ,8  );
    // precompute some factors
    int xsrc = (int)234;//round(x_s/dx);
    int ysrc = (int)round(y_s/dy);
    int zsrc = (int)100;//round(z_s/dz);
    //zeros(Src ,nt,1,1);
load(Src,nt , 1,"Src.dat");
def_sizes(Src,nt,1,1); 
	zeros(Rec ,nt,1,1);
	zeros(Rec_x,nt,1,1);
	zeros(Rec_y,nt,1,1);
	zeros(Rec_y2,nt,1,1);

    DAT xsrcd = (DAT)round(x_s/dx);
    //DAT dt = 4.214791810114839e-06;

    printf("\nxsrc = %50.48d",xsrc);
    printf("\nxsrcd = %50.48f",xsrcd);
    printf("\ndt = %50.48f",dt);
    printf("\nf0 = %50.48f",f0);
    printf("\nx_s = %50.48f",x_s);
    printf("\ndx = %50.48f",dx);
	printf("\nxsrc = %d",xsrc);
	printf("\nzsrc = %d",zsrc);
	printf("\n ");


    // Initial condition
    int isave=0; //  ON
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);
    cudaEventRecord(startD);
    //pert(nt, f0, A0, dt, Src_h);
    //push(Src ,nt,1,1);
    // Initial conditions
    //init<<<grid,block>>>(x_d, y_d, z_d, Prf_d, sigma_xx_d, sigma_yy_d, sigma_zz_d, Vx_d, Vy_d, Qxft_d, Qyft_d, dx, dy,dz, Lx, Ly,Lz,lamx, lamy,lamz, nx, ny, nz); cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        //if (it==51){ tic(); }
        compute_StressPrf<<<grid,block>>>(Prf_d, sigma_xx_d, sigma_yy_d,sigma_zz_d, sigma_xy_d, sigma_xz_d, sigma_yz_d, Vx_d, Vy_d,Vz_d, Qxft_d, Qyft_d,Qzft_d, c11u_het_d,c12u_het_d, weights_x_d, weights_y_d, weights_z_d, weights_xy_d, weights_xz_d, weights_yz_d, dx, dy, dz, dt, c11u,c22u, c33u, c12u,c13u,c23u,         c44u,c55u,c66u, alpha1,alpha2, alpha3, M1, nx, ny, nz);
        cudaDeviceSynchronize();

BC1<<<grid,block>>>(x_d, y_d, z_d, Prf_d, sigma_xx_d, sigma_yy_d, sigma_zz_d, sigma_xy_d, sigma_xz_d, sigma_yz_d, Vx_d, Vy_d, Vz_d, Qxft_d, Qyft_d, Rec_d, dx, dy,dz, Lx, Ly,Lz,lamx, lamy,lamz, nx, ny, nz, it, xsrc, ysrc, zsrc, ds,dt); cudaDeviceSynchronize();

source<<<grid,block>>>(x_d, y_d, z_d, Prf_d, sigma_xx_d, sigma_yy_d, sigma_zz_d, sigma_xy_d,sigma_xz_d, sigma_yz_d, Vx_d, Vy_d, Qxft_d, Qyft_d, Src_d, dx, dy,dz, Lx, Ly,Lz,lamx, lamy,lamz, nx, ny, nz, it, xsrc, ysrc, zsrc, ds,dt); cudaDeviceSynchronize();

        //swap(Qxold, Qxft, tmp11); swap(Qyold, Qyft, tmp22); swap(Qzold, Qzft, tmp33); 
        //cudaDeviceSynchronize();

        update_Qxft<<<grid,block>>>(Prf_d, sigma_xx_d, sigma_yy_d,sigma_zz_d, sigma_xy_d, sigma_xz_d, sigma_yz_d, Vx_d, Vy_d,Vz_d, Qxft_d, Qyft_d,Qzft_d,Qxold_d,Qyold_d,Qzold_d,vrhox11het_d, weights_x_d, weights_y_d, weights_z_d, dx, dy,dz, dt,eta_k1, eta_k2,eta_k3, vrho_x_11, vrho_y_11,vrho_z_11, vrho_x_12,vrho_y_12, vrho_z_12, vrho_x_22,vrho_y_22, vrho_z_22, nx, ny, nz, chi);
        cudaDeviceSynchronize();

record<<<grid,block>>>(x_d, y_d, z_d, Prf_d, sigma_xx_d, sigma_yy_d, sigma_zz_d, Vx_d, Vy_d,Vz_d, Qxft_d, Qyft_d, Rec_d,Rec_x_d,Rec_y_d,Rec_y2_d, dx, dy,dz, Lx, Ly,Lz,lamx, lamy,lamz, nx, ny, nz, it, xsrc, ysrc, zsrc, ds,dt); cudaDeviceSynchronize();

if ((it%nout)==1 && (it > 199)){
printf("\nit=%05d > ", it); fflush(stdout);
}
    }
    //tim("Performance", mem*(nt-50)*42); // timer
    printf("Process %d used GPU with id %d.\n", me, gpu_id);
    cudaEventRecord(stopD);
    cudaEventSynchronize(stopD);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startD, stopD);
    GPUinfo[0] = 1E-3 * milliseconds;
    GPUinfo[1] = it / GPUinfo[0];
    GPUinfo[2] = (it * sizeof(DAT) / (1024 * 1024 * 1024 * GPUinfo[0]));
    printf("\n-----------------------------------");
    printf("\nGPU summary: MTPeff = %.2f [GB/s]", GPUinfo[2]);
    printf("\n-----------------------------------");
    printf("\n  time is %.2f s\n  after %d iterations \n  i.e., %.2f it/s\n", GPUinfo[0], it, GPUinfo[1]);

    ///////////================================================================================ POSTPROCESS ====////
    save_info();  // Save simulation infos and coords (.inf files)
    //SaveArray(x, "xx")
    //SaveArray(y, "yy")
    //SaveArray(z, "zz")
    SaveArray(Src ,"Src");
	SaveArray(Rec ,"Rec");
	SaveArray(Rec_x ,"Rec_x");
	SaveArray(Rec_y ,"Rec_y");
	SaveArray(Rec_y2 ,"Rec_y2");
    SaveArray(Vx   , "Vx"  )
    SaveArray(Vy   , "Vy"  )
    SaveArray(Vz   , "Vz"  )
    //SaveArray(sigma_xx   , "sigma_xx"  )
    //SaveArray(sigma_yy   , "sigma_yy"  )
    //SaveArray(sigma_zz   , "sigma_zz"  )
//SaveArray(vrhox11het   , "vrho_x_11_hsav"  )
    // clear host memory & clear device memory
    free_all(x);
    free_all(y);
    free_all(z);
    free_all(Vx);
    free_all(Vy);
    free_all(Vz);
    free_all(sigma_xx);
    free_all(sigma_yy);
    free_all(sigma_zz);
    free_all(sigma_xy);
    free_all(sigma_xz);
    free_all(sigma_yz);
    free_all(Prf);
    free_all(Qxft);
    free_all(Qyft);
    free_all(Qzft);
    free_all(Qxold);
    free_all(Qyold);
    free_all(Qzold);
    clean_cuda();
    return 0;
}
