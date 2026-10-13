"""hip backend configuration (runner arrives in M14)."""

from kernelport.backends.base import BackendConfig

CONFIG = BackendConfig(
    name="hip",
    extension=".hip",
    description="HIP C++ kernels on AMD GPUs",
    dtypes=('float32', 'float16', 'bfloat16'),
    tolerances={},
    timing_comparable=True,
    timing_note="",
    hardware_hint="AMD GPU with ROCm",
)
