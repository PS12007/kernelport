"""metal (MLX custom Metal kernels) backend configuration."""

from kernelport.backends.base import BackendConfig

PORT_TEMPLATE = '''"""Metal (MLX) port of <kernel>: <one line of what it computes>."""

# Kernel BODY only: MLX writes the signature. Inputs = input/inout tensors then scalars
# (scalars are 1-element arrays: n[0]); outputs = output params, inouts as <name>_out.
SOURCE = """
    uint i = thread_position_in_grid.x;      // CUDA: blockIdx.x * blockDim.x + threadIdx.x
    if (i < uint(n[0])) {
        y[i] = x[i];
    }
"""
THREADGROUP = (256, 1, 1)                     # CUDA: block


def grid(n):
    """TOTAL threads per dimension (Metal dispatchThreads), not threadgroup counts."""
    return (n, 1, 1)
'''

# No tolerance overrides. Metal's compiler may use fast-math, so ports call
# metal::precise:: versions of exp/log/sqrt/rsqrt (see skills/kernelport/backends/metal.md).
CONFIG = BackendConfig(
    name="metal",
    extension=".py",
    description="Metal kernels through MLX custom kernels (mx.fast.metal_kernel)",
    dtypes=("float32", "float16", "bfloat16"),
    tolerances={},
    timing_comparable=False,
    timing_note="wall-clock time including MLX graph evaluation; not comparable to device-event timings",
    hardware_hint="Apple Silicon Mac",
    port_template=PORT_TEMPLATE,
)
