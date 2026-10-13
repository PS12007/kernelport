"""Human-readable hardware names, recorded with every result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


@lru_cache(maxsize=1)
def cpu_name() -> str:
    """CPU model with logical thread count, e.g. 'AMD Ryzen 7 7840HS (16 threads)'."""
    model = " ".join(_cpu_model().split()) or "unknown CPU"
    threads = os.cpu_count()
    return f"{model} ({threads} threads)" if threads else model


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
