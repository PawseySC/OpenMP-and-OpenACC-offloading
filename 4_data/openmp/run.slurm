#!/bin/bash --login 
#SBATCH --reservation=CurtinHPCcourse2026-gpu
#SBATCH --account=courses0100-gpu
#SBATCH --tasks=1 
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --time=00:05:00 
#SBATCH --export=NONE
module load rocm/6.3.0
module load craype-accel-amd-gfx90a
# Compile
echo "Compiling code with clang"
srun --export=all -u -n 1 make -f makefile.clang clean
srun --export=all -u -n 1 make -f makefile.clang
echo "Compiling code with flang"
srun --export=all -u -n 1 make -f makefile.flang clean
srun --export=all -u -n 1 make -f makefile.flang
# Run
echo
echo "Running the c code"
srun --export=all -u -n 1 rocprof --hsa-trace -o c_prof.csv ./laplace 4000
echo "Running the Fortran code"
srun --export=all -u -n 1 rocprof --hsa-trace -o f_prof.csv ./laplacef 4000
