// Embedding lookup: out[t, :] = table[ids[t], :].
// One block per token; threads copy the row with a stride loop.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void embedding_gather(const long long* __restrict__ ids, const T* __restrict__ table,
                                 T* __restrict__ out, int tokens, int dim) {
    int t = blockIdx.x;
    const T* src = table + ids[t] * (long long)dim;
    T* dst = out + (size_t)t * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) dst[d] = src[d];
}
