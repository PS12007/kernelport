"""Capability check for the webgpu backend."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("playwright") is None:
        return False, "playwright not installed (pip install kernelport[webgpu])"
    return False, "webgpu runner not implemented yet (M12)"


def hardware_name() -> str:
    """Filled in by the runner from the WebGPU adapter info."""
    return "unknown"
