#!/bin/bash
#SBATCH --job-name=base_small
#SBATCH --output=base_small_%j.out
#SBATCH --qos=students_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=1
export OMP_PLACES=cores

echo "=== BASELINE: parallel code, 1 MPI proc, 1 OMP thread ==="

echo "Test_02* layer_size=30000:"
for _ in {0..5}
do
    mpirun -np 1 \
        ./energy_storms_mpi_omp 30000 \
        test_files/test_02*
done

echo -e "\n"
echo "Test_personal_01* layer_size=60000:"
for _ in {0..5}
do
    mpirun -np 1 \
        ./energy_storms_mpi_omp 60000 \
        test_files/test_personal_01*
done

echo -e "\n"
echo "Test_personal_02* layer_size=120000:"
for _ in {0..5}
do
    mpirun -np 1 \
        ./energy_storms_mpi_omp 120000 \
        test_files/test_personal_02*
done
