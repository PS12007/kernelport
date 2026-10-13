"""webgpu (WGSL) backend configuration."""

from kernelport.backends.base import BackendConfig

PORT_TEMPLATE = '''"""WebGPU (WGSL) port of <kernel>: <one line of what it computes>."""

# The runner appends: alias T, one storage buffer per tensor param (named like the
# launch.json param), and `params` (a uniform struct holding every scalar param).
WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;                      // CUDA: blockIdx.x * blockDim.x + threadIdx.x
    if (i < u32(params.n)) {            // CUDA: if (i < n)
        y[i] = x[i];
    }
}
"""


def grid(n):
    """Workgroups per dimension (CUDA: the grid). Each must be <= 65535."""
    return ((n + 255) // 256,)
'''

# No tolerance overrides: WGSL f32 add/mul are IEEE single precision. exp/log/sqrt
# precision is implementation-defined in WGSL (exp: 3 + 2*|x| ulp), so ports that
# need tighter results compute those in a numerically safe form; nothing is loosened.
CONFIG = BackendConfig(
    name="webgpu",
    extension=".py",
    description="WGSL compute shaders, run natively through wgpu-native (core WebGPU features only)",
    dtypes=("float32", "float16"),
    tolerances={},
    timing_comparable=False,
    timing_note="wall-clock time including WebGPU submission; not comparable to native kernel timings",
    hardware_hint="any GPU with Vulkan, Metal or D3D12",
    port_template=PORT_TEMPLATE,
)
