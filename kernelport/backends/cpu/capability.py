"""Capability check for the cpu (Numba) backend."""

from __future__ import annotations

import importlib.util
import platform


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("numba") is None:
        return False, "numba not installed (pip install kernelport[cpu])"
    return False, "cpu runner not implemented yet (M11)"


def hardware_name() -> str:
    """CPU description."""
    return platform.processor() or platform.machine() or "CPU"
