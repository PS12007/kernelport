"""hip backend configuration."""

from kernelport.backends.base import BackendConfig

PORT_TEMPLATE = '''// HIP port of <kernel>: <one line of what it computes>.
// Same entry name, parameter order and launch (grid/block/shared mem) as launch.json.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template <typename T>
__global__ void vector_add(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
'''

# No tolerance overrides: HIP float arithmetic is IEEE single precision like CUDA.
CONFIG = BackendConfig(
    name="hip",
    extension=".hip",
    description="HIP C++ kernels on AMD GPUs, compiled at runtime with hipRTC",
    dtypes=("float32", "float16", "bfloat16"),
    tolerances={},
    timing_comparable=True,
    timing_note="",
    hardware_hint="AMD GPU with ROCm (ROCm torch + cupy-rocm)",
    port_template=PORT_TEMPLATE,
)
