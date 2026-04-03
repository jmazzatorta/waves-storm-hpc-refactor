#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "energy_storms.h"

#define CUDA_CHECK_RETURN(value) {                           \
    cudaError_t _m_cudaStat = value;                         \
    if (_m_cudaStat != cudaSuccess) {                        \
        fprintf(stderr, "Error %s at line %d in file %s\n",  \
                cudaGetErrorString(_m_cudaStat),             \
                __LINE__, __FILE__);                         \
        exit(1);                                             \
    }                                                        \
}

#define BLOCK_SIZE 512
#define TILE_SIZE 1024

// Pre-calculated shared memory size for kernels 
#define STENCIL_SHARED_SIZE ((BLOCK_SIZE + 2) * sizeof(float))
#define REDUCE_SHARED_SIZE  (BLOCK_SIZE * sizeof(unsigned long long))

// Max particle number for each storm
#define MAX_STORM_PARTICLES 20000

__device__ __forceinline__ unsigned long long join_values(float val, unsigned int pos) {
    return ((unsigned long long)__float_as_uint(val) << 32) | pos;
}

__global__ void storm_kernel(int num_particles, const int * __restrict__ posval, float * __restrict__ layer, int layer_size, float scaled_layer_size)
{
    // Shared declaration : current tile particles (value and position)
    __shared__ float s_energy_scaled[TILE_SIZE];
    __shared__ int s_pos[TILE_SIZE];

    // Ids calculations
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // Fetch : pre-calc layer value
    float total_energy = 0.0f;
    if (global_id < layer_size) {
        total_energy = layer[global_id];
    }

    // Stride loop : tiles
    for (int tile_start = 0; tile_start < num_particles; tile_start += TILE_SIZE) {

        // Particle num calculation
        int tile_size = num_particles - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Scaling loop : precalculating particles values
        for (int i = tid; i < tile_size; i += BLOCK_SIZE) {

            // Starting point
            int base = (tile_start + i) * 2;

            s_pos[i] = __ldg(&posval[base]);

            float energy = (float)__ldg(&posval[base + 1]);
            s_energy_scaled[i] = __fdiv_rn(energy, scaled_layer_size);
        }

        __syncthreads();

        // Check : out of bound
        if (global_id < layer_size) {

            // Bombing loop
            #pragma unroll 64
            for (int t = 0; t < tile_size; t++) {
                int dist = s_pos[t] - global_id;
                if (dist < 0) dist = -dist;

                float atenuacion = __fsqrt_rn((float)(dist + 1));
                float energy_k = __fdiv_rn(s_energy_scaled[t], atenuacion);

                total_energy += energy_k;
            }
        }
        __syncthreads();
    }

    // Write : post bombing layer value
    if (global_id < layer_size) {
        layer[global_id] = total_energy;
    }
}

__global__ void stencil_kernel(float * __restrict__ layer_r, float * __restrict__ layer_w, int layer_size) {

    // Shared declaration
    extern __shared__ float shared_layer[];

    // Ids calculation
    int global_id = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int tid = threadIdx.x;

    // Check : out of bounds
    if (global_id < layer_size - 1) {

        // Fetch : layer data
        shared_layer[tid + 1] = layer_r[global_id];

        // Fetch if left border
        if (tid == 0 && global_id > 0) {
            shared_layer[0] = layer_r[global_id - 1];
        }

        // Fetch if right border
        if (tid == blockDim.x - 1 || global_id == layer_size - 2) {
            shared_layer[tid + 2] = layer_r[global_id + 1];
        }

        __syncthreads();

        // Stencil
        float local_sum = shared_layer[tid] + shared_layer[tid + 1] + shared_layer[tid + 2];
        layer_w[global_id] = __fdiv_rn(local_sum, 3.0f);
    }
}

__global__ void reduce_max_kernel(float *layer, int layer_size, unsigned long long *d_global_max) {

    // Shared declaration
    extern __shared__ unsigned long long shared_max_pos[];

    // Ids and step calculation
    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    float local_max = 0.0f;
    unsigned int local_pos = 0;

    // Stride loop before reduce
    for (unsigned int i = global_id; i < layer_size; i += step) {

        // Check : valid range
        if (i > 0 && i < (unsigned int)(layer_size - 1)) {

            // Local max
            if (layer[i] > local_max) {
                local_max = layer[i];
                local_pos = i;
            }
        }
    }

    // Packing local max value and pos
    shared_max_pos[tid] = join_values(local_max, local_pos);
    __syncthreads();

    // Reduce max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (tid < s) {

            if (shared_max_pos[tid + s] > shared_max_pos[tid]) {
                shared_max_pos[tid] = shared_max_pos[tid + s];
            }
        }
        __syncthreads();
    }

    // Max outside blocks (and SMs)
    if (tid == 0) {
        atomicMax(d_global_max, shared_max_pos[0]);
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    // GPU check
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA compatible GPU exists.\n");
        return;
    }

    // Grid size calculation
    int grid_size = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocation and cleaning : layer arrays 
    float *d_layer, *d_layer_copy;
    CUDA_CHECK_RETURN(cudaMalloc(&d_layer, layer_size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_layer_copy, layer_size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemset(d_layer, 0, layer_size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemset(d_layer_copy, 0, layer_size * sizeof(float)));

    // Precalculations
    float f_layer_size = (float)layer_size;
    float scaled_layer_size = f_layer_size / 1000.0f;

    // Allocation: packed posval array
    unsigned long long *d_global_max_pos;
    CUDA_CHECK_RETURN(cudaMalloc(&d_global_max_pos, sizeof(unsigned long long)));

    // Allocation: particles and positions array 
    int *d_posval;
    CUDA_CHECK_RETURN(cudaMalloc(&d_posval, MAX_STORM_PARTICLES * 2 * sizeof(int)));

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Main loop: storms
    for (int i = 0; i < num_storms; i++) {

        // Data transfer (host -> device) : storm posvals
        int num_particles = storms[i].size;
        CUDA_CHECK_RETURN(cudaMemcpy(d_posval, storms[i].posval, num_particles * 2 * sizeof(int), cudaMemcpyHostToDevice));

        // Bombing phase
        storm_kernel<<<grid_size, BLOCK_SIZE>>>(num_particles, d_posval, d_layer, layer_size, scaled_layer_size);

        // Rimosso sync inutile

        // Relaxation phase
        stencil_kernel<<<grid_size, BLOCK_SIZE, STENCIL_SHARED_SIZE>>>(&d_layer[1], &d_layer_copy[1], layer_size - 2);

        // Array switch
        float *temp = d_layer;
        d_layer = d_layer_copy;
        d_layer_copy = temp;

        // Prima c'era una memset normale
        
        // Cleaning : packed posvals
        cudaMemsetAsync(d_global_max_pos, 0, sizeof(unsigned long long));

        // Find maximum phase
        reduce_max_kernel<<<grid_size, BLOCK_SIZE, REDUCE_SHARED_SIZE>>>(d_layer, layer_size, d_global_max_pos);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Data transfer (device -> host) : maximum found + position
        unsigned long long global_max_pos;
        CUDA_CHECK_RETURN(cudaMemcpy(&global_max_pos, d_global_max_pos, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // Unpacking posval
        unsigned int raw_max = (unsigned int)(global_max_pos >> 32);
        float global_max;
        memcpy(&global_max, &raw_max, sizeof(float));

        // Write position and value in memory
        maximum[i] = global_max;
        positions[i] = (int)(global_max_pos & 0xFFFFFFFF);
    }

    CUDA_CHECK_RETURN(cudaFree(d_posval));
    CUDA_CHECK_RETURN(cudaFree(d_global_max_pos));
    CUDA_CHECK_RETURN(cudaFree(d_layer));
    CUDA_CHECK_RETURN(cudaFree(d_layer_copy));
}
