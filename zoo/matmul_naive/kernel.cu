// Naive matrix multiply: C[M, N] = A[M, K] @ B[K, N], all row-major.
// One thread per output element; accumulates in float.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void matmul_naive(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                             int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += (float)A[(size_t)row * K + k] * (float)B[(size_t)k * N + col];
        C[(size_t)row * N + col] = (T)acc;
    }
}
