"""Capability check for the webgpu backend (wgpu-native via the `wgpu` package)."""

from __future__ import annotations

import importlib.util


def is_available() -> tuple[bool, str]:
    """(available, reason)."""
    if importlib.util.find_spec("wgpu") is None:
        return False, "wgpu not installed (pip install kernelport[webgpu])"
    if importlib.util.find_spec("torch") is None:
        return False, "torch not installed (needed to build inputs and run references)"
    try:
        from kernelport.backends.webgpu import runner

        name = runner.adapter_name()
    except Exception as exc:
        return False, f"no usable WebGPU adapter: {type(exc).__name__}: {exc}"
    return True, f"wgpu-native on {name}"


def supported_dtypes() -> tuple[str, ...]:
    """float16 only when the adapter offers the standard shader-f16 feature."""
    from kernelport.backends.webgpu import runner

    return ("float32", "float16") if runner.has_f16() else ("float32",)


def hardware_name() -> str:
    """Adapter name and native API, e.g. 'NVIDIA GeForce RTX 5060 Laptop GPU (WebGPU/Vulkan)'."""
    from kernelport.backends.webgpu import runner

    return runner.adapter_name()
