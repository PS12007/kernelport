// SAXPY, in place: y[i] = a * x[i] + y[i]. Computes in float.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void saxpy(int n, float a, const T* __restrict__ x, T* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (T)(a * (float)x[i] + (float)y[i]);
    }
}
