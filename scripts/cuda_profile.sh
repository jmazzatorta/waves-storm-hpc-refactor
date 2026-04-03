#!/bin/bash
#SBATCH --job-name=cuda_profile
#SBATCH --output=cuda_profile_%j.out
#SBATCH --qos=students_limit
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

echo "=============================================="
echo "PROFILING CUDA — ncu (metrics) + nsys (trace)"
echo "=============================================="

# ---- ncu: kernel metrics (occupancy, throughput, FLOP) ----

echo ""
echo "=== NCU METRICS: test_02 (30K) ==="
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    ./energy_storms_cuda 30000 \
    test_files/test_02*

echo ""
echo "=== NCU METRICS: test_07 (1M) ==="
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    ./energy_storms_cuda 1000000 \
    test_files/test_07*

# ---- nsys: timeline trace (H2D / kernel / D2H breakdown) ----

echo ""
echo "=== NSYS TRACE: test_02 (30K) ==="
nsys profile --stats=true -o /tmp/nsys_30k \
    ./energy_storms_cuda 30000 \
    test_files/test_02*

echo ""
echo "=== NSYS TRACE: test_07 (1M) ==="
nsys profile --stats=true -o /tmp/nsys_1m \
    ./energy_storms_cuda 1000000 \
    test_files/test_07*
