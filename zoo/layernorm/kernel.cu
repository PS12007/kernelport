// LayerNorm over the last dim of a [rows, cols] matrix:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// One block per row. Two-pass (mean, then variance of x - mean) in float.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK 256

__device__ float block_sum(float v, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float r = smem[0];
    __syncthreads();  // smem is reused by the next call
    return r;
}

template <typename T>
__global__ void layernorm(const T* __restrict__ x, const T* __restrict__ gamma,
                          const T* __restrict__ beta, T* __restrict__ y,
                          int rows, int cols, float eps) {
    __shared__ float smem[BLOCK];
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;

    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) s += (float)xr[c];
    float mean = block_sum(s, smem) / cols;

    float sq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float d = (float)xr[c] - mean;
        sq += d * d;
    }
    float rstd = rsqrtf(block_sum(sq, smem) / cols + eps);

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float v = ((float)xr[c] - mean) * rstd;
        yr[c] = (T)(v * (float)gamma[c] + (float)beta[c]);
    }
}
