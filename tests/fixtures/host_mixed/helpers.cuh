#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define DEVICE_INLINE __device__ __forceinline__
#define WARP 32

cublasHandle_t global_handle;          // host global: dropped
static size_t workspace_size = 1 << 20; // host global: dropped

DEVICE_INLINE float twice(float x) { return 2.0f * x; }

template<class T>
__host__ __device__ T cdiv(T a, T b) { return (a + b - 1) / b; }

void host_only_helper(cudaStream_t s) {  // host function: dropped
    cudaStreamSynchronize(s);
}
