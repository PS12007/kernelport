// HIP port of transpose: same algorithm as kernel.cu; CUDA headers swapped for HIP ones.
// Matrix transpose: out[c][r] = in[r][c] for a [rows, cols] row-major input.
// Classic 32x32 shared-memory tile with +1 padding to avoid bank conflicts.
// Block is 32x8: each thread moves 4 elements of the tile.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define TILE 32
#define ROWS_PER_PASS 8

template <typename T>
__global__ void transpose(const T* __restrict__ in, T* __restrict__ out, int rows, int cols) {
    __shared__ T tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;  // column in input
    int y = blockIdx.y * TILE + threadIdx.y;  // row in input
    for (int j = 0; j < TILE; j += ROWS_PER_PASS) {
        if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;  // column in output (= input row)
    y = blockIdx.x * TILE + threadIdx.y;  // row in output (= input column)
    for (int j = 0; j < TILE; j += ROWS_PER_PASS) {
        if (x < rows && y + j < cols) out[(size_t)(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
