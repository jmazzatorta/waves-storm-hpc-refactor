#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define ALIGN_SIZE        64

// Safe for unrolling (avx256 8 float)
#define SIMD_WIDTH        16

#define CACHE_LINE_FLOATS 16

// Maximum storms width + 2
#define STORM_WIDTH       30002

typedef struct {
    float value;
    int position;
} MaxInfo;

#pragma omp declare reduction(maxinfo : MaxInfo : \
    omp_out = (omp_in.value > omp_out.value) ? omp_in : omp_out) \
    initializer(omp_priv = {0.0f, -1})

// Manual scheduling based on cache line sized chunks
#define THREAD_VALUES(n, t_start, t_end)                                    \
    int t_id           = omp_get_thread_num();                              \
    int n_threads      = omp_get_num_threads();                             \
    int total_chunks   = ((n) + CACHE_LINE_FLOATS - 1) / CACHE_LINE_FLOATS; \
    int chunks_per_thr = total_chunks / n_threads;                          \
    int rem_chunks     = total_chunks % n_threads;                          \
    int my_chunks      = chunks_per_thr + (t_id < rem_chunks ? 1 : 0);      \
    int start_chunk    = t_id * chunks_per_thr + MIN(t_id, rem_chunks);     \
    int t_start        = start_chunk * CACHE_LINE_FLOATS;                   \
    int t_end          = MIN(t_start + my_chunks * CACHE_LINE_FLOATS, (n))

static inline float* alloc_aligned(size_t count) {
    size_t padded = ((count + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    return (float *)aligned_alloc(ALIGN_SIZE, padded * sizeof(float));
}

static float *g_sqrt_table = NULL;
static int table_size = 0;

static void init_sqrt_table(int size) {

    table_size = size;
    g_sqrt_table = alloc_aligned(size);
    g_sqrt_table[0] = 0.0f;

    #pragma omp parallel for simd schedule(static)
    for (int d = 1; d < size; d++) {
        g_sqrt_table[d] = sqrtf((float)d);
    }
}

static void free_sqrt_table(void) {
    free(g_sqrt_table);
    g_sqrt_table = NULL;
}

static inline void init_buffers(float * restrict layer, float * restrict aux, int local_count) {

    #pragma omp parallel
    {
        THREAD_VALUES(local_count, t_start, t_end);

        // First touch policy
        #pragma omp simd
        for (int i = t_start; i < t_end; i++) {
            layer[i] = 0.0f;
            aux[i]   = 0.0f;
        }
    }
}

static void compute_storm(const Storm *storm, float scaled_layer_size, float * restrict local_layer, int offset, int local_count) {

    // Restricted arrays
    const int num_particles       = storm->size;
    const int * restrict posval   = storm->posval;
    const float * restrict sqrt_table = g_sqrt_table;

    #pragma omp parallel
    {
        THREAD_VALUES(local_count, t_start, t_end);

        // Bombing loop
        for (int p = 0; p < num_particles; p++) {

            // Fetch posval
            int pos   = posval[p * 2];
            float val = (float)posval[p * 2 + 1] / scaled_layer_size;

            // Calculates pos in MPI chunk
            int local_pos = pos - offset;

            // Restrict area if particle out of OMP chunk(s)
            int split = local_pos;
            if (split < t_start) split = t_start;
            if (split > t_end) split = t_end;

            // Loop : particle left side 
            if (split > t_start) {
                #pragma omp simd
                for (int i = t_start; i < split; i++) {
                    local_layer[i] += val / sqrt_table[local_pos - i + 1];
                }
            }

            // Loop : particle left side 
            if (t_end > split) {
                #pragma omp simd
                for (int i = split; i < t_end; i++) {
                    local_layer[i] += val / sqrt_table[i - local_pos + 1];
                }
            }
        }
    }
}

static void relax_and_find_maximum(float * restrict local_layer, float * restrict aux_layer, int local_count, int offset, int layer_size, int left_rank, int right_rank, MPI_Request *requests, float *out_max_val, int *out_max_idx) {

    if (local_count <= 0) return;

    float left_halo = 0.0f, right_halo = 0.0f;

    // Halo exchange 
    MPI_Isend(&local_layer[0], 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&local_layer[local_count - 1], 1, MPI_FLOAT, right_rank, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(&left_halo, 1, MPI_FLOAT, left_rank, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&right_halo, 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, &requests[3]);

    if (local_count > 2) {

        #pragma omp parallel
        {
            THREAD_VALUES(local_count, t_start, t_end);
            int loop_start = MAX(t_start, 1);
            int loop_end   = MIN(t_end, local_count - 1);

            if (loop_end > loop_start) {

                // Stencil inside OMP chunk
                #pragma omp simd
                for (int i = loop_start; i < loop_end; i++) {
                    aux_layer[i] = (local_layer[i - 1] + local_layer[i] + local_layer[i + 1]) / 3.0f;
                }
            }
        }
    }

    // Vediamo se hanno finito quelli di sopra
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // Left border stencil
    {
        float l = (offset == 0) ? 0.0f : left_halo;
        float r = (local_count > 1) ? local_layer[1] : 0.0f;
        aux_layer[0] = (l + local_layer[0] + r) / 3.0f;
    }

    // Right border stencil
    if (local_count > 1) {
        int last = local_count - 1;
        float l  = local_layer[last - 1];
        float r  = (offset + last == layer_size - 1) ? 0.0f : right_halo;
        aux_layer[last] = (l + local_layer[last] + r) / 3.0f;
    }

    MaxInfo result = {0.0f, -1};

    // Find maximum phase
    #pragma omp parallel reduction(maxinfo : result)
    {
        THREAD_VALUES(local_count, t_start, t_end);

        for (int i = t_start; i < t_end; i++) {

            float current = aux_layer[i];
            int global_k  = offset + i;

            if (global_k > 0 && global_k < layer_size - 1) {
                if (current > result.value) {
                    result.value    = current;
                    result.position = i;
                }
            }

            // Copy che mi evito il memcpy
            local_layer[i] = current;
        }
    }

    *out_max_val = result.value;
    *out_max_idx = result.position;
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, int rank, int size) {

    // Partitioning layer chunks for MPI ranks
    int base_chunk  = layer_size / size;
    int remainder   = layer_size % size;
    int local_count = base_chunk + (rank < remainder ? 1 : 0);
    int offset      = rank * base_chunk + MIN(rank, remainder);

    // Allocation layer arrays
    float *local_layer = alloc_aligned(local_count);
    float *aux_layer   = alloc_aligned(local_count);
    init_buffers(local_layer, aux_layer, local_count);

    // Precalculation
    const float scaled_layer_size = (float)layer_size / 1000.0f;

    // Square root lookup table
    init_sqrt_table(STORM_WIDTH);

    // Declaring requests for non-blocking halos exchange
    MPI_Request requests[4];

    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Main loop : storms
    for (int s = 0; s < num_storms; s++) {

        // Bombing phase
        compute_storm(&storms[s], scaled_layer_size, local_layer, offset, local_count);

        // Merged function : relaxation phase + find maximum phase
        float local_max_val;
        int   local_max_idx;
        relax_and_find_maximum(local_layer, aux_layer, local_count, offset, layer_size, left_rank, right_rank, requests, &local_max_val, &local_max_idx);

        // Structure for reduction (local maximum value, mpi rank)
        struct { float val; int rank; } local_max, global_max;
        local_max.val  = local_max_val;
        local_max.rank = rank;

        // Allreduce (every rank receives maximum value)
        MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == 0) {

            // Write maximum value
            maximum[s] = global_max.val;

            // If 0 is the biggest writes position, or waits for the winner 
            if (global_max.rank == 0) {
                positions[s] = (local_max_idx != -1) ? (offset + local_max_idx) : -1;

            } else {
                MPI_Recv(&positions[s], 1, MPI_INT, global_max.rank, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        } else if (rank == global_max.rank) {

            // If rank has the maximum value sends position to rank 0 
            int global_pos = (local_max_idx != -1) ? (offset + local_max_idx) : -1;
            MPI_Send(&global_pos, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }
    }

    free_sqrt_table();
    free(aux_layer);
    free(local_layer);
}
