#!/bin/bash
#SBATCH --job-name=mpi_2node_waves
#SBATCH --output=mpi_2node_waves_%j.out
#SBATCH --qos=students_limit
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Test_02* layer_size=30000:"
for _ in {0..5}
do
    mpirun -np 32 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        --oversubscribe \
        --mca coll ^hcoll \
        ./energy_storms_mpi_omp 30000 \
        test_files/test_02*
done

echo -e "\n"
echo "Test_personal_01* layer_size=60000:"
for _ in {0..5}
do
    mpirun -np 32 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        --oversubscribe \
        --mca coll ^hcoll \
        ./energy_storms_mpi_omp 60000 \
        test_files/test_personal_01*
done

echo -e "\n"
echo "Test_personal_02* layer_size=120000:"
for _ in {0..5}
do
    mpirun -np 32 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        --oversubscribe \
        --mca coll ^hcoll \
        ./energy_storms_mpi_omp 120000 \
        test_files/test_personal_02*
done

echo -e "\n"
echo "Test_07* layer_size=1000000:"
for _ in {0..5}
do
    mpirun -np 32 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        --oversubscribe \
        --mca coll ^hcoll \
        ./energy_storms_mpi_omp 1000000 \
        test_files/test_07*
done

echo -e "\n"
echo "Test_personal_03* layer_size=2000000:"
for _ in {0..5}
do
    mpirun -np 32 \
        --map-by ppr:2:numa:PE=2 \
        --bind-to core \
        --oversubscribe \
        --mca coll ^hcoll \
        ./energy_storms_mpi_omp 2000000 \
        test_files/test_personal_03*
done
