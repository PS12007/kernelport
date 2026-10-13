"""metal backend configuration (runner arrives in M13)."""

from kernelport.backends.base import BackendConfig

CONFIG = BackendConfig(
    name="metal",
    extension=".py",
    description="MLX custom Metal kernels (mx.fast.metal_kernel)",
    dtypes=('float32', 'float16', 'bfloat16'),
    tolerances={},
    timing_comparable=True,
    timing_note="",
    hardware_hint="Apple Silicon Mac",
)
