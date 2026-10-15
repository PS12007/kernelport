"""Deterministic input generation from a launch spec.

Values come from NumPy's PCG64 ``random_raw`` stream, whose bit output is
guaranteed stable across NumPy versions and platforms. Uniform and normal
values are derived from those raw bits here (not via NumPy's distribution
methods, which are allowed to change), so any machine regenerates
bit-identical inputs from the same spec.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from kernelport.harness.expr import evaluate
from kernelport.harness.spec import FLOAT_DTYPES, CaseSpec, LaunchSpec, ParamSpec

TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

INT_SENTINEL = -0x5A5A5A5  # fills int outputs so unwritten elements are visible
DEFAULT_EDGE_BIG = 1000.0
NUMERIC_INIT_KEYS = ("low", "high", "mean", "std", "edge_big", "step")


@dataclass
class CaseData:
    """Concrete tensors and scalars for one (case, dtype)."""

    case: str
    dtype: str
    dims: dict[str, Any]
    tensors: dict[str, torch.Tensor]  # CPU tensors, inputs and pre-filled outputs
    scalars: dict[str, Any]
    input_hash: str

    @property
    def label(self) -> str:
        """Short name such as ``odd_1000/float16``."""
        return f"{self.case}/{self.dtype}"


def _uniform01(gen: np.random.Generator, n: int) -> np.ndarray:
    """n doubles in [0, 1): 52 raw bits placed in the mantissa of [1, 2), minus 1. Exact."""
    raw = gen.bit_generator.random_raw(n)
    raw >>= np.uint64(12)
    raw |= np.uint64(0x3FF0000000000000)
    out = raw.view(np.float64)
    out -= 1.0
    return out


def _normal(gen: np.random.Generator, n: int) -> np.ndarray:
    """n values with mean 0 and std 1, bell-shaped (Irwin-Hall: sum of 4 uniforms).

    Only uses IEEE add/multiply, which are exactly rounded on every platform.
    Box-Muller would need log/sin/cos, whose last bit differs between libms.
    Values are bounded to about (-3.46, 3.46), which is fine for test inputs.
    """
    s = _uniform01(gen, n)
    for _ in range(3):  # elementwise adds in a fixed order
        s += _uniform01(gen, n)
    s -= 2.0
    s *= 1.7320508075688772  # sqrt(3): var of the sum of 4 uniforms is 1/3
    return s


def _seed_for(spec: LaunchSpec, case: CaseSpec, param: str) -> int:
    # Same values for every dtype of a case, so dtypes are comparable.
    return (spec.seed * 1_000_003 + zlib.crc32(f"{case.name}|{param}".encode())) & 0xFFFFFFFF


def _edge_values(big: float) -> list[float]:
    return [0.0, -0.0, 1.0, -1.0, big, -big, 1.0 / big, -1.0 / big]


def _generate(init: dict, shape: tuple[int, ...], dtype: str, seed: int) -> torch.Tensor:
    is_float = dtype in FLOAT_DTYPES
    vals = _base_values(json.dumps(init, sort_keys=True), shape, is_float, seed)
    return torch.from_numpy(vals.copy()).to(TORCH_DTYPES[dtype])


@lru_cache(maxsize=4)  # dtypes of one case share values; keep the last few
def _base_values(init_json: str, shape: tuple[int, ...], is_float: bool, seed: int) -> np.ndarray:
    """float64 values before the dtype cast."""
    init = json.loads(init_json)
    n = int(np.prod(shape)) if shape else 1
    gen = np.random.Generator(np.random.PCG64(seed))
    dist = init.get("dist", "normal" if is_float else "randint")
    if dist == "normal":
        vals = init.get("mean", 0.0) + init.get("std", 1.0) * _normal(gen, n)
    elif dist == "uniform":
        lo, hi = init.get("low", -1.0), init.get("high", 1.0)
        vals = lo + (hi - lo) * _uniform01(gen, n)
    elif dist == "randint":
        lo, hi = int(init.get("low", 0)), int(init.get("high", 100))
        vals = np.floor(lo + (hi - lo) * _uniform01(gen, n))
    elif dist == "zeros":
        vals = np.zeros(n)
    elif dist == "ones":
        vals = np.ones(n)
    elif dist == "arange":
        vals = np.arange(n, dtype=np.float64)
    elif dist == "full":  # e.g. a running max that starts at -inf: {"dist": "full", "value": "-inf"}
        if "value" not in init:
            raise ValueError('dist "full" needs a "value" (a number, "inf", "-inf" or "nan")')
        vals = np.full(n, float(init["value"]))
    else:
        raise ValueError(f"unknown dist {dist!r}")
    if init.get("edges") and is_float:
        edges = _edge_values(float(init.get("edge_big", DEFAULT_EDGE_BIG)))[:n]
        vals[: len(edges)] = edges
    step = init.get("step")
    if step and is_float:
        # Snap to multiples of a power-of-two step. Sums of such values are exact
        # in float32 (while partial sums stay below 2**24 * step), so reductions
        # give the same answer in any order and strict tolerances still apply.
        if not math.log2(step).is_integer():
            raise ValueError(f"step must be a power of two, got {step}")
        vals = np.round(vals / step) * step
    vals = np.ascontiguousarray(vals.reshape(shape))
    vals.flags.writeable = False
    return vals


def _sentinel(shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    if dtype in FLOAT_DTYPES:
        return torch.full(shape, float("nan"), dtype=TORCH_DTYPES[dtype])
    return torch.full(shape, INT_SENTINEL, dtype=torch.int64).to(TORCH_DTYPES[dtype])


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a CPU tensor, for hashing."""
    t = t.detach().cpu().contiguous()
    if t.dim() == 0:
        t = t.reshape(1)
    return t.view(torch.uint8).numpy().tobytes()


def _as_scalar_dtype(value, dtype: str):
    """The value the kernel actually receives: 0.999 as a float32 param is 0.99900001...

    References, originals and ports all get this rounded value, so a reference never
    disagrees with the kernel just because it used the unrounded Python float.
    """
    if dtype in ("float32", "float16"):
        return float(np.dtype(dtype).type(value))
    if dtype in ("int32", "int64", "uint32", "int8", "uint8", "int16"):
        return int(value)
    return value


def make_case(spec: LaunchSpec, case: CaseSpec, dtype: str) -> CaseData:
    """Build CPU tensors for one case/dtype. Outputs are pre-filled with a sentinel."""
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, Any] = {}
    h = hashlib.sha256()
    for p in spec.params:
        if not p.is_tensor:
            scalars[p.name] = _as_scalar_dtype(p.resolved_value(case.dims), p.dtype)
            continue
        shape = p.resolved_shape(case.dims)
        pdtype = p.resolved_dtype(dtype)
        if p.kind == "output":
            t = _sentinel(shape, pdtype)
        else:
            init = {**spec.init.get(p.name, {}), **case.init.get(p.name, {})}
            # Numeric fields may be expressions over dims, e.g. "high": "max_pos".
            init = {k: (evaluate(v, case.dims) if k in NUMERIC_INIT_KEYS and isinstance(v, str) else v)
                    for k, v in init.items()}
            t = _generate(init, shape, pdtype, _seed_for(spec, case, p.name))
        tensors[p.name] = t
        h.update(f"{p.name}:{pdtype}:{shape}".encode())
        h.update(tensor_bytes(t))
    h.update(repr(sorted(scalars.items())).encode())
    return CaseData(case.name, dtype, dict(case.dims), tensors, scalars, h.hexdigest())


def output_params(spec: LaunchSpec) -> list[ParamSpec]:
    """Params the kernel writes."""
    return [p for p in spec.params if p.is_output]
