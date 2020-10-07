#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug 
#SBATCH -J testsimuonly39GHz 
#SBATCH --mail-user=jost@apc.in2p3.fr
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

angle=0
freq=39
simu_number=16
#run the application:
srun -n 16 -c 4 --cpu_bind=cores python /global/homes/j/jost/these/spectra_based_analysis/code/cl_sim_mpi2.py --alpha $angle --Frequency $freq
