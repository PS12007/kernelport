// Kernels defined through macros, mixed with host code. Compiled by NVRTC in the tests.
#include <stdio.h>
#include <cuda_runtime.h>

#define CAT(a, b) a##b
#define KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// a family of kernels, one per type
#define DEFINE_SCALE(T, SUF)                                        \
    __global__ void CAT(scale_, SUF)(T* x, const T s, int n) {      \
        KERNEL_LOOP(i, n) { x[i] = x[i] * s; }                      \
    }
#define FOR_FLOAT_TYPES(M) M(float, f32) M(double, f64)
FOR_FLOAT_TYPES(DEFINE_SCALE)

// a host helper stamped out by a macro, with no trailing semicolon
#define DECLARE_HOST_HELPER(name) int host_##name(int a) { printf("%d\n", a); return a; }
DECLARE_HOST_HELPER(counter)

// a qualifier macro standing for __global__
#define KERNEL __global__ __launch_bounds__(128)
KERNEL void marked_add(const float* a, const float* b, float* out, int n) {
    KERNEL_LOOP(i, n) { out[i] = a[i] + b[i]; }
}

// a macro that opens the definition; the parameters and body follow it
#define C_KERNEL(name) extern "C" __global__ void name
C_KERNEL(opened_relu)(const float* x, float* y, int n) {
    KERNEL_LOOP(i, n) { y[i] = x[i] > 0.0f ? x[i] : 0.0f; }
}

// a plain kernel, for contrast
__global__ void plain_copy(const float* x, float* y, int n) {
    KERNEL_LOOP(i, n) { y[i] = x[i]; }
}

int main() {
    float* d = nullptr;
    cudaMalloc(&d, 1024 * sizeof(float));
    scale_f32<<<4, 256>>>(d, 2.0f, 1024);
    marked_add<<<8, 128>>>(d, d, d, 1024);
    printf("%d\n", host_counter(1));
    return 0;
}
