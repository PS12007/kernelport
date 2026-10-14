"""Run an original CUDA kernel through CuPy (NVRTC). No nvcc or Visual Studio needed.

The kernel source is compiled once per (source, entry list) and launched with
the grid/block from the launch spec. Inputs go in as CPU tensors and outputs
come back as CPU tensors.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kernelport.harness.inputs import CaseData
from kernelport.harness.spec import LaunchSpec
from kernelport.hashing import sha256_file

NVRTC_OPTIONS = ("-std=c++17",)

# nvcc gets these from host math headers; NVRTC does not. Defined only if missing.
NVRTC_PRELUDE = """
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
#line 1
"""

_SCALAR_NP = {
    "float32": np.float32, "float64": np.float64, "float16": np.float16,
    "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
}


def is_available() -> tuple[bool, str]:
    """Can original CUDA kernels run here?"""
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        return False, f"cupy not importable ({exc}); pip install kernelport[cuda]"
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False, "no NVIDIA CUDA GPU visible to torch"
    return True, "ok"


@lru_cache(maxsize=64)
def _compile(source: str, entries: tuple[str, ...], include_dir: str,
             prelude: str = NVRTC_PRELUDE, options: tuple[str, ...] = NVRTC_OPTIONS):
    import cupy as cp

    opts = options + (f"-I{include_dir}",)  # so #include "local.cuh" works
    mod = cp.RawModule(code=prelude + source, options=opts, name_expressions=list(entries))
    return {e: mod.get_function(e) for e in entries}


def get_kernel(source_path: Path, spec: LaunchSpec, dtype: str, *, type_names: dict[str, str] | None = None,
               prelude: str = NVRTC_PRELUDE, options: tuple[str, ...] = NVRTC_OPTIONS,
               device_only: bool = True):
    """Compiled CuPy function for this spec's entry at ``dtype``.

    ``type_names`` maps dtypes to the C++ names substituted for ``{T}`` in the entry
    (CUDA names by default; the hip backend passes HIP names). CuPy compiles with
    NVRTC on NVIDIA and hipRTC on ROCm. With ``device_only`` (the default, for original
    CUDA files) host code is sliced away first, see ``device_source``.
    """
    from kernelport.harness.device_source import device_source

    def entry(t: str) -> str:
        return spec.entry.replace("{T}", type_names[t]) if type_names else spec.entry_for(t)

    if device_only:
        source = device_source(Path(source_path))
    else:
        source = Path(source_path).read_text(encoding="utf-8")
    entries = tuple(sorted({entry(t) for t in spec.dtypes}))
    funcs = _compile(source, entries, str(Path(source_path).resolve().parent), prelude, options)
    return funcs[entry(dtype)]


def _as_cupy(t: torch.Tensor):
    import cupy as cp

    if t.dtype == torch.bfloat16:  # cupy has no bf16; same bits as int16
        t = t.view(torch.int16)
    return cp.asarray(t)


def kernel_args(spec: LaunchSpec, data: CaseData, dev: dict[str, torch.Tensor]) -> list[Any]:
    args: list[Any] = []
    for p in spec.params:
        if p.is_tensor:
            args.append(_as_cupy(dev[p.name]))
        else:
            args.append(_SCALAR_NP[p.resolved_dtype(data.dtype)](data.scalars[p.name]))
    return args


def prepare(source_path: Path, spec: LaunchSpec, data: CaseData):
    """Return (launch_fn, device_tensors). launch_fn() runs the kernel once, async."""
    kern = get_kernel(source_path, spec, data.dtype)
    dev = {k: v.to("cuda") for k, v in data.tensors.items()}
    grid, block, shmem = spec.grid_block(data.dims)
    args = tuple(kernel_args(spec, data, dev))

    def launch() -> None:
        kern(grid, block, args, shared_mem=shmem)

    return launch, dev


def run_original(source_path: Path, spec: LaunchSpec, data: CaseData) -> dict[str, torch.Tensor]:
    """Run once and return output tensors on the CPU."""
    launch, dev = prepare(source_path, spec, data)
    launch()
    torch.cuda.synchronize()
    return {p.name: dev[p.name].cpu() for p in spec.params if p.is_output}


def source_hash(path: Path) -> str:
    """sha256 of the kernel source."""
    return sha256_file(path)
