"""Human-readable hardware names, recorded with every result."""

from __future__ import annotations

import platform


def cpu_name() -> str:
    """Best-effort CPU description."""
    return platform.processor() or platform.machine() or "unknown CPU"


def gpu_name() -> str:
    """Name of GPU 0 as seen by torch, with arch, or '' if none."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    name = torch.cuda.get_device_name(0)
    if getattr(torch.version, "hip", None):
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return f"{name} ({arch.split(':')[0]})" if arch else name
    major, minor = torch.cuda.get_device_capability(0)
    return f"{name} (sm_{major}{minor})"
