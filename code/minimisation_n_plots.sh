#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J minimisation_16sim_280GHz
#SBATCH --mail-user=jost@apc.in2p3.fr
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

angle=0
freq=280
simu_number=16
#run the application:
srun -n 16 -c 4 --cpu_bind=cores python /global/homes/j/jost/these/spectra_based_analysis/code/angle_estimation.py --alpha $angle --Frequency $freq
srun -n 1 -c 1 --cpu_bind=cores python /global/homes/j/jost/these/spectra_based_analysis/code/spectra_plot_script.py --alpha $angle --Frequency $freq --Nsim $simu_number
srun -n 1 -c 1 --cpu_bind=cores python /global/homes/j/jost/these/spectra_based_analysis/code/histo_plot.py --alpha $angle --Frequency $freq --Nsim $simu_number
