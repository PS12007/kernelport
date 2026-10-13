// HIP port of rope: same algorithm as kernel.cu; CUDA headers swapped for HIP ones.
// Rotary position embedding (interleaved pairs, as in the original RoPE).
// For token t at position p = positions[t], head h, pair i (0 <= i < D/2):
//   x0 = x[t,h,2i], x1 = x[t,h,2i+1], c = cos_cache[p,i], s = sin_cache[p,i]
//   y[t,h,2i]   = x0 * c - x1 * s
//   y[t,h,2i+1] = x0 * s + x1 * c
// One block per token; threads stride over (head, pair). Math in float.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template <typename T>
__global__ void rope(const long long* __restrict__ positions, const T* __restrict__ x,
                     const float* __restrict__ cos_cache, const float* __restrict__ sin_cache,
                     T* __restrict__ y, int tokens, int heads, int head_dim) {
    int t = blockIdx.x;
    int half = head_dim / 2;
    long long p = positions[t];
    const float* c_row = cos_cache + p * half;
    const float* s_row = sin_cache + p * half;
    for (int j = threadIdx.x; j < heads * half; j += blockDim.x) {
        int h = j / half, i = j % half;
        size_t base = ((size_t)t * heads + h) * head_dim + 2 * i;
        float x0 = (float)x[base], x1 = (float)x[base + 1];
        float c = c_row[i], s = s_row[i];
        y[base] = (T)(x0 * c - x1 * s);
        y[base + 1] = (T)(x0 * s + x1 * c);
    }
}
