"""Triton backend configuration."""

from kernelport.backends.base import BackendConfig

PORT_TEMPLATE = '''"""Triton port of <kernel>: <one line of what it computes>."""

import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # One program per BLOCK elements (CUDA: one thread block).
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n  # CUDA: if (i < n)
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)


def run(x, y, n):
    """Same parameter names as launch.json. Tensors are on the GPU; write outputs in place."""
    BLOCK = 1024
    _kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
'''

# No tolerance overrides: Triton emits the same IEEE float32 add/mul/fma as CUDA,
# and tl.exp / tl.log lower to libdevice-accurate or ex2/lg2-based sequences whose
# error (a few ulp) is inside the float32 default (rtol 1.3e-6 ~ 11 ulp).
# Note: tl.dot defaults to TF32 for float32 inputs, which is NOT within tolerance;
# ports must pass input_precision="ieee" for float32 matmuls.
CONFIG = BackendConfig(
    name="triton",
    extension=".py",
    description="Triton kernels launched from Python (NVIDIA via triton / triton-windows, AMD via ROCm triton)",
    dtypes=("float32", "float16", "bfloat16"),
    tolerances={},
    timing_comparable=True,
    hardware_hint="NVIDIA GPU (CUDA) or AMD GPU (ROCm)",
    port_template=PORT_TEMPLATE,
    tune_space={"BLOCK": [128, 256, 512, 1024, 2048], "num_warps": [2, 4, 8]},
)
