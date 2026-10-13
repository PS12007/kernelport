// HIP port of relu: same algorithm as kernel.cu; CUDA headers swapped for HIP ones.
// Element-wise ReLU: y[i] = max(x[i], 0).
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template <typename T>
__global__ void relu(const T* __restrict__ x, T* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T v = x[i];
        y[i] = v > T(0.0f) ? v : T(0.0f);
    }
}
