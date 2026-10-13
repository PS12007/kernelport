"""Capability check for the hip backend (ROCm torch + ROCm CuPy with hipRTC)."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed"
    import torch

    if not getattr(torch.version, "hip", None):
        return False, "needs a ROCm build of torch on an AMD GPU"
    if not torch.cuda.is_available():
        return False, "ROCm torch sees no AMD GPU"
    if importlib.util.find_spec("cupy") is None:
        return False, "cupy (ROCm build, e.g. cupy-rocm-6-0) not installed; it provides hipRTC compilation"
    try:
        import cupy

        if not getattr(cupy.cuda.runtime, "is_hip", False):
            return False, "cupy is the CUDA build; install the ROCm build (e.g. cupy-rocm-6-0)"
    except Exception as exc:
        return False, f"cupy failed to import: {exc}"
    return True, f"hipRTC via CuPy on ROCm {torch.version.hip}"


def hardware_name() -> str:
    """AMD GPU name with gfx arch."""
    from kernelport.hardware import gpu_name

    return gpu_name() or "no GPU"
