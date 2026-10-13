"""webgpu backend configuration (runner arrives in M12)."""

from kernelport.backends.base import BackendConfig

CONFIG = BackendConfig(
    name="webgpu",
    extension=".wgsl",
    description="WGSL compute shaders in headless Chrome via Playwright",
    dtypes=('float32', 'float16'),
    tolerances={},
    timing_comparable=False,
    timing_note="browser timings include dispatch overhead and are not comparable to native GPU numbers",
    hardware_hint="any machine with Chrome + a WebGPU-capable GPU",
)
