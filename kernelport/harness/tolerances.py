"""Comparison tolerances. These live in kernelport, never come from the agent.

A value passes when ``|actual - expected| <= atol + rtol * |expected|`` where
``expected`` is the float64 reference rounded to the output dtype. Integer
outputs must match exactly.

Defaults are torch.testing.assert_close's per-dtype defaults, a widely used
standard:
- float32 rtol 1.3e-6, atol 1e-5: ~11 ulp relative slack for reordered
  arithmetic; atol covers values that cancel to ~0.
- float16 rtol 1e-3, atol 1e-5: one fp16 ulp is ~9.8e-4 relative, so this
  allows about one ulp of disagreement after rounding.
- bfloat16 rtol 1.6e-2, atol 1e-5: one bf16 ulp is ~7.8e-3 relative, so this
  allows about two ulps.
- float64 rtol 1e-7, atol 1e-7: as torch.testing.

Backends may override a dtype with a documented reason (backends/*/config.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    """Allowed error for one dtype."""

    rtol: float
    atol: float
    reason: str = ""


DEFAULT_TOLERANCES: dict[str, Tolerance] = {
    "float32": Tolerance(1.3e-6, 1e-5, "torch.testing default for float32"),
    "float16": Tolerance(1e-3, 1e-5, "torch.testing default for float16 (~1 ulp)"),
    "bfloat16": Tolerance(1.6e-2, 1e-5, "torch.testing default for bfloat16 (~2 ulp)"),
    "float64": Tolerance(1e-7, 1e-7, "torch.testing default for float64"),
}

EXACT = Tolerance(0.0, 0.0, "integer outputs must match exactly")


def tolerance_for(dtype: str, overrides: dict[str, Tolerance] | None = None) -> Tolerance:
    """Tolerance for an output dtype, honouring backend overrides."""
    if overrides and dtype in overrides:
        return overrides[dtype]
    return DEFAULT_TOLERANCES.get(dtype, EXACT)
