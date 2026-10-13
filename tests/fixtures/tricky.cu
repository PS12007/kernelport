// Fixture for analyzer tests. Not compiled.
// A comment mentioning __global__ void fake_in_comment(int x) { } must be ignored.
#include <cuda_fp16.h>

/* block comment:
   __global__ void fake_in_block_comment(float* p) { p[0] = 1; }
*/
const char* banner = "__global__ void fake_in_string(int) {}";

__global__ void declared_only(float* out, int n);

__device__ __forceinline__ float warp_reduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

template <typename scalar_t, int BLOCK = 256, typename Acc = float>
__global__ void __launch_bounds__(BLOCK, 2)
multi_line_kernel(
    const scalar_t* __restrict__ input,   // read only
    scalar_t* __restrict__ output,        // written
    float* partial,                       // atomically updated
    const int64_t* idx,
    int n,
    float scale)
{
    extern __shared__ float smem[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t* out_row = output + blockIdx.y * n;
    float v = i < n ? (float)input[idx[i]] * scale : 0.0f;
    v = warp_reduce(v);
    if (threadIdx.x % 32 == 0) atomicAdd(&partial[blockIdx.x], v);
    __syncthreads();
    if (i < n) out_row[i] = (scalar_t)v;
}

extern "C" __global__ void plain_c(float4* data, const half2* h, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < count; i += blockDim.x * gridDim.x) {
        float4 d = data[i];
        d.x += 1.0f;
        data[i] = d;
    }
}

__global__ void overloaded(float* a) { a[0] = 1.0f; }
__global__ void overloaded(double* a) { *a = 2.0; }

__global__ void passes_to_helper(float* buf, unsigned long long* counter) {
    helper(buf);
    counter[0]++;
}

void host_launch(float* d_out, int n, cudaStream_t s) {
    dim3 grid((n + 255) / 256), block(256);
    multi_line_kernel<float, 256><<<grid, block, 256 * sizeof(float), s>>>(d_in, d_out, d_part, d_idx, n, 0.5f);
    plain_c<<<(n + 127) / 128, 128>>>(d_data, d_h, n);
}
