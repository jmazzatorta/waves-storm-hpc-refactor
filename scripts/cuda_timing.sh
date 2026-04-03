#!/bin/bash
#SBATCH --job-name=cuda_waves
#SBATCH --output=cuda_waves_%j.out
#SBATCH --qos=students_limit
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

echo "Test_02* layer_size=30000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 30000 \
        test_files/test_02*
done

echo -e "\n"
echo "Test_personal_01* layer_size=60000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 60000 \
        test_files/test_personal_01*
done

echo -e "\n"
echo "Test_personal_02* layer_size=120000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 120000 \
        test_files/test_personal_02*
done

echo -e "\n"
echo "Test_07* layer_size=1000000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 1000000 \
        test_files/test_07*
done

echo -e "\n"
echo "Test_personal_03* layer_size=2000000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 2000000 \
        test_files/test_personal_03*
done

echo -e "\n"
echo "Test_personal_04* layer_size=4000000:"
for _ in {0..5}
do
    srun -N 1 -n 1 ./energy_storms_cuda 4000000 \
        test_files/test_personal_04*
done
