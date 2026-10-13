// HIP port of cumsum: same algorithm as kernel.cu; CUDA headers swapped for HIP ones.
// Inclusive prefix sum along each row of a [rows, cols] matrix:
//   y[r, c] = x[r, 0] + ... + x[r, c]
// One block per row. The row is processed in BLOCK-sized chunks; each chunk
// is scanned in shared memory (Hillis-Steele) and offset by the running
// total of the previous chunks. Accumulates in float.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define BLOCK 256

template <typename T>
__global__ void cumsum(const T* __restrict__ x, T* __restrict__ y, int rows, int cols) {
    __shared__ float buf[BLOCK];
    __shared__ float carry;
    const T* xr = x + (size_t)blockIdx.x * cols;
    T* yr = y + (size_t)blockIdx.x * cols;
    int tid = threadIdx.x;
    if (tid == 0) carry = 0.0f;
    __syncthreads();

    for (int start = 0; start < cols; start += BLOCK) {
        int c = start + tid;
        buf[tid] = c < cols ? (float)xr[c] : 0.0f;
        __syncthreads();
        // Hillis-Steele inclusive scan of buf.
        for (int off = 1; off < BLOCK; off <<= 1) {
            float add = tid >= off ? buf[tid - off] : 0.0f;
            __syncthreads();
            buf[tid] += add;
            __syncthreads();
        }
        if (c < cols) yr[c] = (T)(carry + buf[tid]);
        __syncthreads();
        if (tid == 0) carry += buf[BLOCK - 1];
        __syncthreads();
    }
}
