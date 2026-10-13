"""Run a metal port: a Metal kernel body compiled by MLX (``mx.fast.metal_kernel``).

A metal port is a Python file defining::

    SOURCE = '''
        uint i = thread_position_in_grid.x;
        if (i < n[0]) { out[i] = a[i] + b[i]; }
    '''
    THREADGROUP = (256, 1, 1)
    def grid(n):                 # TOTAL threads per dimension (MLX/Metal dispatchThreads),
        return (n, 1, 1)         # not threadgroup counts as in CUDA

MLX writes the kernel signature from these names (launch.json order):

- inputs: every ``input`` and ``inout`` tensor param, then every scalar param as a
  one-element array (read it as ``n[0]``);
- outputs: every ``output`` param by name, and every ``inout`` param as ``<name>_out``
  (MLX outputs are fresh buffers: copy or update from the input ``<name>``).

``T`` is a template parameter when launch.json uses dtype "T". Outputs are
pre-filled with ``INIT_VALUE`` (default NaN, so unwritten elements are caught).
``ATOMIC_OUTPUTS = True`` declares every output as ``device atomic<...>*``.
``HEADER`` may hold helper functions placed before the kernel.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from kernelport.backends.base import Prepared, bind_kwargs
from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import file_hash, load_module
from kernelport.harness.spec import LaunchSpec

_loaded: dict[tuple[str, str], object] = {}
_kernels: dict[tuple, object] = {}
SCALAR_NP = {"int32": np.int32, "int64": np.int64, "uint32": np.uint32, "float32": np.float32,
             "float64": np.float32, "float16": np.float16}


def _mx():
    import mlx.core as mx

    return mx


def mx_dtype(dtype: str):
    mx = _mx()
    return {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16, "int32": mx.int32,
            "int64": mx.int64, "uint32": mx.uint32, "int8": mx.int8, "uint8": mx.uint8,
            "int16": mx.int16}[dtype]


def load_port(path: Path):
    path = Path(path).resolve()
    key = (str(path), file_hash(path))
    if key not in _loaded:
        mod = load_module(path, "kp_metal")
        if not isinstance(getattr(mod, "SOURCE", None), str):
            raise AttributeError(f"{path.name} must define SOURCE = '''...''' (the Metal kernel body)")
        if not callable(getattr(mod, "grid", None)):
            raise AttributeError(f"{path.name} must define grid(...) returning total threads per dimension")
        _loaded[key] = mod
    return _loaded[key]


def names(spec: LaunchSpec) -> tuple[list[str], list[str]]:
    """(input_names, output_names) as the kernel sees them."""
    ins = [p.name for p in spec.params if p.kind in ("input", "inout")]
    ins += [p.name for p in spec.params if p.kind == "scalar"]
    outs = [p.name if p.kind == "output" else f"{p.name}_out" for p in spec.params if p.is_output]
    return ins, outs


def _to_mx(t: torch.Tensor):
    mx = _mx()
    if t.dtype == torch.bfloat16:  # NumPy has no bf16: go through float32 (exact both ways)
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.contiguous().numpy())


def _to_torch(a, dtype: str) -> torch.Tensor:
    mx = _mx()
    if dtype == "bfloat16":
        return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a).copy())


def _triple(v, what: str) -> tuple[int, int, int]:
    v = (v,) if isinstance(v, int) else tuple(int(x) for x in v)
    if not 1 <= len(v) <= 3 or any(x < 0 for x in v):
        raise ValueError(f"{what} must be 1 to 3 non-negative ints, got {v}")
    return v + (1,) * (3 - len(v))


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Build the MLX kernel (cached) and bind it to one case."""
    mx = _mx()
    mod = load_port(path)
    in_names, out_names = names(spec)
    atomic = bool(getattr(mod, "ATOMIC_OUTPUTS", False))
    header = getattr(mod, "HEADER", "") or ""
    key = (mod.SOURCE, header, tuple(in_names), tuple(out_names), atomic)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"kp_{spec.name}", input_names=in_names, output_names=out_names,
            source=mod.SOURCE, header=header, ensure_row_contiguous=True, atomic_outputs=atomic)
    kernel = _kernels[key]

    inputs = [_to_mx(data.tensors[p.name]) for p in spec.params if p.kind in ("input", "inout")]
    for p in spec.params:
        if p.kind == "scalar":
            inputs.append(mx.array(np.array([data.scalars[p.name]], dtype=SCALAR_NP[p.dtype])))
    outs = [p for p in spec.params if p.is_output]
    out_dtypes = [p.resolved_dtype(data.dtype) for p in outs]
    out_shapes = [tuple(data.tensors[p.name].shape) for p in outs]
    template = [("T", mx_dtype(data.dtype))] if any(p.dtype == "T" for p in spec.params) else []

    arrays = {p.name: data.tensors[p.name].numpy() if data.tensors[p.name].dtype != torch.bfloat16
              else data.tensors[p.name] for p in spec.params if p.is_tensor}
    g = _triple(mod.grid(**bind_kwargs(mod.grid, spec, arrays, data.scalars)), "grid()")
    tg = _triple(getattr(mod, "THREADGROUP", (256, 1, 1)), "THREADGROUP")
    init = getattr(mod, "INIT_VALUE", math.nan)
    state: dict = {}

    def launch() -> None:
        state["outs"] = kernel(inputs=inputs, template=template, grid=g, threadgroup=tg,
                               output_shapes=out_shapes, output_dtypes=[mx_dtype(d) for d in out_dtypes],
                               init_value=init, verbose=False)

    def sync() -> None:
        if "outs" in state:
            mx.eval(*state["outs"])

    def fetch() -> dict:
        return {p.name: _to_torch(a, d) for p, a, d in zip(outs, state["outs"], out_dtypes)}

    return Prepared(launch=launch, sync=sync, fetch=fetch, timer="wall_clock")
