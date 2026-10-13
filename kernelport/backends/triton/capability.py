"""Can Triton ports run here? Imports nothing heavy at module level."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed (pip install kernelport[triton])"
    if importlib.util.find_spec("triton") is None:
        return False, "triton not installed (pip install kernelport[triton]; triton-windows on Windows)"
    import torch

    if not torch.cuda.is_available():
        return False, "no CUDA or ROCm GPU visible to torch"
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"triton failed to import: {exc}"
    return True, "Triton on " + ("ROCm" if getattr(torch.version, "hip", None) else "CUDA")


def hardware_name() -> str:
    """GPU name with arch, e.g. 'NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)'."""
    from kernelport.hardware import gpu_name

    return gpu_name() or "no GPU"
