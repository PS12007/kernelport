"""cpu (Numba) backend configuration."""

from kernelport.backends.base import BackendConfig

PORT_TEMPLATE = '''"""CPU (Numba) port of <kernel>: <one line of what it computes>."""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _kernel(x, y, n):
    # CUDA: one thread per i. prange splits the loop across CPU threads.
    for i in prange(n):
        y[i] = x[i]


def run(x, y, n):
    """Same parameter names as launch.json. Arrays are NumPy with launch.json shapes; write outputs in place."""
    _kernel(x, y, n)
'''

# No tolerance overrides: Numba compiles float32 arithmetic to IEEE float32 ops
# (fastmath is off unless a port turns it on), and math.exp/log/sqrt on float32
# call the C library's correctly-rounded-or-near versions.
CONFIG = BackendConfig(
    name="cpu",
    extension=".py",
    description="Numba-compiled CPU kernels (any machine)",
    dtypes=("float32", "float64"),
    tolerances={},
    timing_comparable=False,
    timing_note="CPU wall-clock time; not comparable to GPU kernel timings",
    hardware_hint="any machine with Numba",
    port_template=PORT_TEMPLATE,
)
