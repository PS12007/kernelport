"""Capability check for the metal (MLX) backend."""

from __future__ import annotations

import importlib.util
import platform
import sys


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False, "Metal needs macOS on Apple Silicon"
    if importlib.util.find_spec("mlx") is None:
        return False, "mlx not installed (pip install kernelport[metal])"
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed (needed to build inputs and run references)"
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return False, "MLX reports no Metal device"
    except Exception as exc:
        return False, f"mlx failed to import: {exc}"
    return True, "MLX custom Metal kernels"


def hardware_name() -> str:
    """Apple chip name, e.g. 'Apple M2 Pro (Metal)'."""
    from kernelport.hardware import _cpu_model

    return f"{_cpu_model() or 'Apple Silicon'} (Metal)"
