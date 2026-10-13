// Element-wise vector addition: out[i] = a[i] + b[i].
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void vector_add(const T* __restrict__ a, const T* __restrict__ b,
                           T* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
