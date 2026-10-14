// A real-world style .cu file: kernels mixed with host code, as nvcc sees it.
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include "helpers.cuh"

#define SCALE (M_PI / 4.0f)

typedef float (*unary_fn)(float);
using index_t = int;
const int kBlock = 256;
__constant__ float bias_table[4] = {0.0f, 1.0f, 2.0f, 3.0f};
std::vector<int> host_sizes(4);
int host_counter = 0;

struct alignas(8) Pair {
    float a, b;
    __device__ float sum() const { return a + b; }
};

enum class Mode : int { Add = 0, Mul = 1 };

void cpu_reference(float* out, const float* in, int n);  // host prototype: dropped
__device__ float clamp_max(float x);

namespace detail {
__device__ float clamp_max(float x) { return fminf(x, FLT_MAX); }
int host_in_namespace(int x) { return x + 1; }
}

__device__ float clamp_max(float x) { return detail::clamp_max(x); }

void cpu_reference(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) out[i] = in[i] * SCALE;
    printf("done %d\n", n);
}

extern "C" __global__ void scale_kernel(float* out, const float* in, int n) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = clamp_max(twice(in[i]) * SCALE * 0.5f + bias_table[0]);
}

extern "C" {
__global__ void pair_kernel(float* out, const Pair* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i].sum();
}
int host_in_extern_c(int x) { return x; }
}

template<int N>
__global__ void unrolled_kernel(float* out, int n) {
    int i = threadIdx.x;
    if (i < n) out[i] = (float)N * cdiv(i, WARP);
}

int main(int argc, char** argv) {
    int n = 1 << 20;
    float *d_out, *d_in;
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_in, n * sizeof(float));
    scale_kernel<<<cdiv(n, kBlock), kBlock>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    return 0;
}
