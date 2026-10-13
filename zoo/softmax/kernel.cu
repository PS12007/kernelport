// Row-wise softmax over a [rows, cols] row-major matrix.
// One block per row; each thread strides over the columns. Accumulates in
// float, uses the max-subtraction trick for numerical stability.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK 256

template <typename T>
__global__ void softmax(const T* __restrict__ x, T* __restrict__ y, int rows, int cols) {
    __shared__ float red[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const T* xr = x + (size_t)row * cols;
    T* yr = y + (size_t)row * cols;

    // 1) row max
    float m = -INFINITY;
    for (int c = tid; c < cols; c += BLOCK) m = fmaxf(m, (float)xr[c]);
    red[tid] = m;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    // 2) sum of exp(x - max)
    float sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK) sum += expf((float)xr[c] - m);
    red[tid] = sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];

    // 3) normalise
    for (int c = tid; c < cols; c += BLOCK) yr[c] = (T)(expf((float)xr[c] - m) * inv);
}
