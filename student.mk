# --- Silence ---
MAKEFLAGS += -s

# --- Execution Parameters ---
export OMP_NUM_THREADS=4
MPI_PROCS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) --oversubscribe \
               --map-by ppr:2:numa:PE=2 \
               --bind-to core \
			   --mca coll ^hcoll

# --- Compiler Flags ---
MPI_OMP_EXTRA_CFLAGS = -O3 -march=znver1 -mtune=znver1 \
                       -mfma -mavx2 -fno-math-errno -fno-trapping-math \
                       -fno-signaling-nans -ffinite-math-only \
                       -fno-signed-zeros \
                       -funroll-loops -ftree-vectorize -flto \
                       -fstrict-aliasing \
                       -fprefetch-loop-arrays \
                       --param l1-cache-size=32 \
                       --param l1-cache-line-size=64 \
                       --param l2-cache-size=512 \
                       -fno-stack-protector -fno-plt

MPI_OMP_EXTRA_LIBS = -flto

CUDA_EXTRA_CFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3
CUDA_EXTRA_LIBS =
