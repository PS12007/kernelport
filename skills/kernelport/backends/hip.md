# Porting CUDA to HIP (AMD GPUs)

HIP is AMD's CUDA-like C++ dialect: most CUDA kernels port by swapping headers and a
few intrinsics. A hip port is a `.hip` source file that kernelport compiles at runtime
with hipRTC (through CuPy's ROCm build) and **launches exactly like the original**:
same entry name (`launch.json` `entry`, `{T}` filled with HIP type names), same
parameter order, same grid, block and shared memory.

Verification needs an AMD GPU with ROCm torch and ROCm CuPy (e.g. `cupy-rocm-6-0`). On
other machines `verify_port` returns UNVERIFIED. **Status: the runner's launch path is
exercised on NVIDIA in tests, but hipRTC compilation and the zoo ports have not yet been
verified on AMD hardware.**

If you only need AMD support, a Triton port also runs on ROCm unchanged; HIP is for
staying close to the CUDA source.

## Port file format

```cpp
// HIP port of vector_add.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template <typename T>
__global__ void vector_add(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
```

- Keep the kernel name, template parameters and parameter order of the original.
  The launch comes from launch.json, so the kernel must work with the original
  grid/block (loop inside the kernel if you need a different mapping).
- `{T}` becomes `float`, `__half` or `__hip_bfloat16`.

## CUDA → HIP mapping

| CUDA | HIP |
|---|---|
| `#include <cuda_fp16.h>` / `<cuda_bf16.h>` | `#include <hip/hip_fp16.h>` / `<hip/hip_bf16.h>` |
| `__nv_bfloat16` | `__hip_bfloat16` |
| `blockIdx`, `threadIdx`, `__syncthreads`, `__shared__`, `atomicAdd` | same |
| `__shfl_xor_sync(0xffffffff, v, off)` | `__shfl_xor(v, off)`. **Wavefronts are 64 lanes on CDNA (MI-series), 32 on RDNA**: use `warpSize`, never a hard-coded 32 |
| warp-level reductions sized for 32 lanes | a shared-memory tree reduction (wave-size independent), or loops over `warpSize` |
| `__ballot_sync` / masks (32-bit) | `__ballot` returns a 64-bit mask |
| `cudaStream_t`, host API | not used: kernelport does the launch |

## Bugs checklist

- [ ] No `0xffffffff`-masked `*_sync` intrinsics and no hard-coded warp size 32.
- [ ] Shared arrays sized by the block size, not by warp counts that assume 32 lanes.
- [ ] Same entry name, template parameters and parameter order as launch.json.
- [ ] HIP headers and type names (`__hip_bfloat16`).
