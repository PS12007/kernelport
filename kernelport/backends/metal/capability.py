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
    return False, "metal runner not implemented yet (M13)"


def hardware_name() -> str:
    """Apple chip name if known."""
    return platform.processor() or "Apple Silicon"
