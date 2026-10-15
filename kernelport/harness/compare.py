"""Compare a backend's output against the frozen reference.

Checks, in order: shape, dtype, NaN placement, Inf placement (with sign),
then element-wise tolerance. The failure report is compact so it fits in an
agent's context: one line of numbers plus small expected/actual slices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from kernelport.harness.tolerances import Tolerance

SLICE_RADIUS = 4  # elements shown either side of the worst index


@dataclass
class TensorCompare:
    """Result of comparing one output tensor."""

    name: str
    passed: bool
    shape: list[int]
    dtype: str
    reason: str = ""  # empty on pass
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    worst_index: list[int] | None = None
    num_mismatched: int = 0
    num_elements: int = 0
    expected_slice: list[Any] = field(default_factory=list)
    actual_slice: list[Any] = field(default_factory=list)
    slice_start: int = 0  # flat index of the first slice element

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return asdict(self)

    def summary(self) -> str:
        """One-line human summary."""
        if self.passed:
            return f"{self.name}: ok (max abs {self.max_abs_err:.3g}, max rel {self.max_rel_err:.3g})"
        return f"{self.name}: {self.reason}"


def _dtype_name(t: torch.dtype) -> str:
    return str(t).replace("torch.", "")


def _fmt(v: float) -> Any:
    if v != v or v in (float("inf"), float("-inf")):
        return str(v)
    return float(f"{v:.6g}")


def _slices(expected: torch.Tensor, actual: torch.Tensor, flat_idx: int) -> tuple[list, list, int]:
    e, a = expected.reshape(-1), actual.reshape(-1)
    lo = max(0, flat_idx - SLICE_RADIUS)
    hi = min(e.numel(), flat_idx + SLICE_RADIUS + 1)
    return [_fmt(x) for x in e[lo:hi].tolist()], [_fmt(x) for x in a[lo:hi].tolist()], lo


def _unravel(flat: int, shape: tuple[int, ...]) -> list[int]:
    idx = []
    for s in reversed(shape):
        idx.append(flat % s if s else 0)
        flat //= s if s else 1
    return list(reversed(idx))


def compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor,
                   out_dtype: torch.dtype, tol: Tolerance) -> TensorCompare:
    """Compare ``actual`` (backend output) to ``expected`` (reference, any precision)."""
    res = TensorCompare(name, False, list(expected.shape), _dtype_name(out_dtype),
                        num_elements=expected.numel())
    actual = actual.detach().cpu()
    if tuple(actual.shape) != tuple(expected.shape):
        res.reason = f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
        return res
    if actual.dtype != out_dtype:
        res.reason = f"dtype mismatch: expected {_dtype_name(out_dtype)}, got {_dtype_name(actual.dtype)}"
        return res
    if expected.numel() == 0:
        res.passed = True
        return res

    if not out_dtype.is_floating_point:
        exp_i = expected.to(out_dtype).to(torch.int64)
        act_i = actual.to(torch.int64)
        bad = exp_i != act_i
        diff = (exp_i - act_i).abs()
        res.max_abs_err = float(diff.max())
        res.num_mismatched = int(bad.sum())
        if res.num_mismatched:
            flat = int(torch.argmax(bad.reshape(-1).to(torch.int8)))
            res.worst_index = _unravel(flat, tuple(expected.shape))
            res.expected_slice, res.actual_slice, res.slice_start = _slices(exp_i, act_i, flat)
            res.reason = (f"{res.num_mismatched}/{res.num_elements} integer values differ; "
                          f"first at {res.worst_index}")
            return res
        res.passed = True
        return res

    # Round the exact reference to the output dtype, then compare in float64.
    exp = expected.to(out_dtype).double()
    act = actual.double()

    exp_nan, act_nan = torch.isnan(exp), torch.isnan(act)
    if not torch.equal(exp_nan, act_nan):
        bad = exp_nan != act_nan
        flat = int(torch.argmax(bad.reshape(-1).to(torch.int8)))
        res.num_mismatched = int(bad.sum())
        res.worst_index = _unravel(flat, tuple(exp.shape))
        res.expected_slice, res.actual_slice, res.slice_start = _slices(exp, act, flat)
        unexpected = bool(act_nan.reshape(-1)[flat])
        hint = (" (an element never written stays NaN; inf/inf or inf-inf, e.g. from an overflowing exp, "
                "also gives NaN)") if unexpected else ""
        res.reason = (f"{'unexpected' if unexpected else 'missing'} NaN: {res.num_mismatched}/{res.num_elements} "
                      f"NaN positions differ; first at {res.worst_index}{hint}")
        return res

    exp_inf, act_inf = torch.isinf(exp), torch.isinf(act)
    inf_bad = (exp_inf | act_inf) & ~((exp_inf & act_inf) & (torch.sign(exp) == torch.sign(act)))
    if bool(inf_bad.any()):
        flat = int(torch.argmax(inf_bad.reshape(-1).to(torch.int8)))
        res.num_mismatched = int(inf_bad.sum())
        res.worst_index = _unravel(flat, tuple(exp.shape))
        res.expected_slice, res.actual_slice, res.slice_start = _slices(exp, act, flat)
        res.reason = f"Inf mismatch: {res.num_mismatched}/{res.num_elements} positions; first at {res.worst_index}"
        return res

    finite = torch.isfinite(exp)
    e, a = torch.where(finite, exp, 0.0), torch.where(finite, act, 0.0)
    abs_err = (a - e).abs()
    rel_err = abs_err / e.abs().clamp_min(torch.finfo(torch.float64).tiny)
    allowed = tol.atol + tol.rtol * e.abs()
    excess = abs_err - allowed
    res.max_abs_err = float(abs_err.max())
    res.max_rel_err = float(torch.where(abs_err > 0, rel_err, 0.0).max())
    bad = excess > 0
    res.num_mismatched = int(bad.sum())
    if res.num_mismatched:
        flat = int(torch.argmax(excess.reshape(-1)))
        res.worst_index = _unravel(flat, tuple(exp.shape))
        res.expected_slice, res.actual_slice, res.slice_start = _slices(exp, act, flat)
        res.reason = (f"{res.num_mismatched}/{res.num_elements} values outside tolerance "
                      f"(rtol={tol.rtol:g}, atol={tol.atol:g}); max abs err {res.max_abs_err:.4g}, "
                      f"max rel err {res.max_rel_err:.4g}, worst at {res.worst_index}")
        return res
    res.passed = True
    return res
