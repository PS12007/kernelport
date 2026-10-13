"""Capability check for the hip backend."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed"
    import torch

    if not getattr(torch.version, "hip", None):
        return False, "needs a ROCm build of torch on an AMD GPU"
    return False, "hip runner not implemented yet (M14)"


def hardware_name() -> str:
    """AMD GPU name with gfx arch."""
    from kernelport.hardware import gpu_name

    return gpu_name() or "no GPU"
