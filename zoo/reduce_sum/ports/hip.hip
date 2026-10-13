// HIP port of reduce_sum: same algorithm as kernel.cu; CUDA headers swapped for HIP ones.
// Full sum reduction: *out += sum(x[0..n)). Accumulates in float.
// Each block reduces a grid-stride slice in shared memory, then thread 0
// adds the block's partial into *out with atomicAdd. *out must start at 0.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define BLOCK 256

template <typename T>
__global__ void reduce_sum(const T* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = blockIdx.x * BLOCK + tid; i < n; i += gridDim.x * BLOCK) acc += (float)x[i];
    smem[tid] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}
