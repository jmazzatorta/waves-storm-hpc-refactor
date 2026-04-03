#!/bin/bash
#SBATCH --job-name=mpi_prof_base
#SBATCH --output=mpi_prof_base_%j.out
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

echo "=== PROFILING: baseline 1 proc, 1 thread, test_07 (1M) ==="

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,instructions,cycles,task-clock \
    mpirun -np 1 \
        ./energy_storms_mpi_omp 1000000 \
        test_files/test_07*
