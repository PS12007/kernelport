"""cpu backend configuration (runner arrives in M11)."""

from kernelport.backends.base import BackendConfig

CONFIG = BackendConfig(
    name="cpu",
    extension=".py",
    description="Numba-compiled CPU kernels",
    dtypes=('float32', 'float64'),
    tolerances={},
    timing_comparable=False,
    timing_note="CPU timings are not comparable to GPU numbers",
    hardware_hint="any machine",
)
