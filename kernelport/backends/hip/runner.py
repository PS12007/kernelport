"""Run a hip port: HIP C++ compiled at runtime with hipRTC (through CuPy's ROCm build).

A hip port is a ``.hip`` source file containing the kernel(s). It is launched
exactly like the original CUDA kernel: same entry (``launch.json`` "entry",
with ``{T}`` filled from HIP type names), same parameter order, same grid,
block and shared memory from ``launch.json``. Tensors are device buffers on the
AMD GPU (outputs pre-filled with NaN); scalars are passed by value.

Requirements: a ROCm build of torch and a ROCm build of CuPy (e.g. cupy-rocm-6-0),
which compiles RawModule sources with hipRTC.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kernelport.backends.base import Prepared
from kernelport.harness import run_cuda
from kernelport.harness.inputs import CaseData
from kernelport.harness.spec import LaunchSpec

# C++ names substituted for {T} in the entry. HIP spells bf16 differently from CUDA.
HIP_TYPE_NAMES = {
    "float32": "float", "float64": "double", "float16": "__half", "bfloat16": "__hip_bfloat16",
    "int8": "int8_t", "uint8": "uint8_t", "int16": "int16_t", "int32": "int", "int64": "long long",
}

# hipRTC, like NVRTC, has no host math headers. hip_runtime is implicit in hipRTC.
HIPRTC_PRELUDE = run_cuda.NVRTC_PRELUDE
HIPRTC_OPTIONS = ("-std=c++17",)


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Compile (cached) and bind the port to one case on the GPU."""
    kern = run_cuda.get_kernel(path, spec, data.dtype, type_names=HIP_TYPE_NAMES,
                               prelude=HIPRTC_PRELUDE, options=HIPRTC_OPTIONS, device_only=False)
    dev = {k: v.to("cuda") for k, v in data.tensors.items()}  # torch's ROCm build keeps the "cuda" name
    grid, block, shmem = spec.grid_block(data.dims)
    args = tuple(run_cuda.kernel_args(spec, data, dev))
    outputs = [p.name for p in spec.params if p.is_output]
    return Prepared(
        launch=lambda: kern(grid, block, args, shared_mem=shmem),
        sync=torch.cuda.synchronize,
        fetch=lambda: {n: dev[n].cpu() for n in outputs},
        timer="cuda_event",  # torch.cuda.Event maps to hipEvent on ROCm
    )
