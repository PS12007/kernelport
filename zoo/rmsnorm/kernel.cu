// RMSNorm over the last dim of a [rows, cols] matrix:
//   y = x / sqrt(mean(x^2) + eps) * weight
// One block per row (256 threads = 8 warps). The sum of squares is reduced
// with warp shuffles, then across warps through shared memory.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK 256
#define WARPS (BLOCK / 32)

__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

template <typename T>
__global__ void rmsnorm(const T* __restrict__ x, const T* __restrict__ weight,
                        T* __restrict__ y, int rows, int cols, float eps) {
    __shared__ float warp_partial[WARPS];
    __shared__ float inv_rms;
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

    float sq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float v = (float)xr[c];
        sq += v * v;
    }
    sq = warp_sum(sq);
    if (lane == 0) warp_partial[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        float t = lane < WARPS ? warp_partial[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) inv_rms = rsqrtf(t / cols + eps);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        yr[c] = (T)((float)xr[c] * inv_rms * (float)weight[c]);
    }
}
