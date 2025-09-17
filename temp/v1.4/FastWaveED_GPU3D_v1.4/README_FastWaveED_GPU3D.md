
# FastWaveED_GPU3D

version 1.4

## Software

**FastWaveED_GPU3D** is a GPU-accelerated simulation code for modeling 3D elastic wave propagation during earthquakes. The core solver is written in CUDA C. Input generation, compilation, and postprocessing are handled in MATLAB. The code is optimized for a single NVIDIA GPU and achieves high performance for full-scale earthquake simulations.

26 July 2025  
Copyright (C) 2025  
Yury Alkhimenkov, Massachusetts Institute of Technology

## Usage

- Input: All input files (e.g., material parameters, moment tensor) are generated automatically by the MATLAB script `vizme_FastBiotED_GPU3D_OK_2024_final.m`.

- Output: The code produces seismograms in binary format (e.g., `*_Rec_y2.res`) which are read and visualized by MATLAB.

To run the simulation:

1. Open MATLAB and navigate to the folder `FastWaveED_GPU3D_v1_DC_Earthquake_FULL_OK4_final`.

2. Execute the MATLAB script:
```matlab
vizme_FastBiotED_GPU3D_OK_2024_final
```

This script will:
- Define the model size and simulation parameters
- Generate 3D material property and source files
- Compile the CUDA kernel using a command like:
```sh
nvcc -arch=sm_89 -O3 -DNBX=12 -DNBY=12 -DNZ=7 -DOVERX=0 -DOVERY=0 -DOVERZ=0 -DNPARS1=10 -DNPARS2=13 -DNPARS3=15 FastWaveED_GPU3D_v1.cu
```
- Run the executable:
```sh
a.exe
```
- Read the output binary files and visualize results in MATLAB

### Requirements
- Windows OS  
- MATLAB (must be in system PATH)  
- CUDA Toolkit 10.0 or newer  
- NVIDIA GPU with compute capability 8.9 or higher (e.g., `sm_89`)

## License

FastWaveED_GPU3D is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.

FastWaveED_GPU3D is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  
GNU General Public License for more details.

You should have received a copy of the GNU General Public License  
along with FastWaveED_GPU3D. If not, see <http://www.gnu.org/licenses/>.
