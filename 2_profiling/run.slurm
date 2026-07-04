#!/bin/bash --login 
#SBATCH --reservation=HPCUniofMelbourne
#SBATCH --account=courses01
#SBATCH --tasks=1 
#SBATCH --partition=work 
#SBATCH --time=00:05:00 
#SBATCH --export=NONE
module purge
module load craype-x86-milan
module load PrgEnv-gnu/8.6.0
# Compile
echo "Compiling code"
srun --export=all -u -n 1 make clean
srun --export=all -u -n 1 make 
# Run
echo
echo "Running C code"
srun --export=all -u -n 1 ./laplace 4000
# Generate profiling report for c
echo
echo "Generating profiling report for C"
srun --export=all -u -n 1 gprof -lbp laplace gmon.out
# Run
echo
echo "Running Fortran code"
srun --export=all -u -n 1 ./laplacef 4000
# Generate profiling report for fortran
echo
echo "Generating profiling report for Fortran"
srun --export=all -u -n 1 gprof -lbp laplacef gmon.out
