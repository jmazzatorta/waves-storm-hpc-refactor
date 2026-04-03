#!/bin/bash
#SBATCH --job-name=mpi_prof_1node
#SBATCH --output=mpi_prof_1node_%j.out
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

echo "=== PROFILING: 1 node, 16 ranks x 4 threads, test_07 (1M) ==="

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,instructions,cycles,task-clock \
    mpirun -np 16 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        ./energy_storms_mpi_omp 1000000 \
        test_files/test_07*
