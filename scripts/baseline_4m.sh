#!/bin/bash
#SBATCH --job-name=base_4m
#SBATCH --output=base_4m_%j.out
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

echo "=== BASELINE: parallel code, 1 MPI proc, 1 OMP thread ==="

echo "Test_personal_04* layer_size=4000000:"
for _ in {0..5}
do
    mpirun -np 1 \
        ./energy_storms_mpi_omp 4000000 \
        test_files/test_personal_04*
done
