// HIP port of rmsnorm: y = x / sqrt(mean(x^2) + eps) * weight, one block per row.
// The CUDA kernel reduces with 32-lane warp shuffles; AMD wavefronts are 64 wide on
// CDNA (32 on RDNA), so this port uses a shared-memory tree reduction that does not
// depend on the wavefront size.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define BLOCK 256

template <typename T>
__global__ void rmsnorm(const T* __restrict__ x, const T* __restrict__ weight,
                        T* __restrict__ y, int rows, int cols, float eps) {
    __shared__ float red[BLOCK];
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    int tid = threadIdx.x;

    float sq = 0.0f;
    for (int c = tid; c < cols; c += BLOCK) {
        float v = (float)xr[c];
        sq += v * v;
    }
    red[tid] = sq;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(red[0] / cols + eps);

    for (int c = tid; c < cols; c += BLOCK) {
        yr[c] = (T)((float)xr[c] * inv_rms * (float)weight[c]);
    }
}
