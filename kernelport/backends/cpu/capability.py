"""Capability check for the cpu (Numba) backend."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("numba") is None:
        return False, "numba not installed (pip install kernelport[cpu])"
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed (needed to build inputs and run references)"
    try:
        import numba
    except Exception as exc:  # e.g. numba built against a different NumPy
        return False, f"numba failed to import: {exc}"
    return True, f"Numba {numba.__version__} on CPU, {numba.config.NUMBA_NUM_THREADS} threads"


def hardware_name() -> str:
    """CPU model name, e.g. 'Intel(R) Core(TM) i7-13620H (16 threads)'."""
    from kernelport.hardware import cpu_name

    return cpu_name()
