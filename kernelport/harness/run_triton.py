"""Run a hand-written or agent-written Triton port.

A Triton port is a Python file defining::

    def run(<same parameter names as the CUDA kernel>):
        ...launch one or more @triton.jit kernels...

Tensor parameters arrive as device tensors (outputs pre-filled with NaN, or a
sentinel for ints) and must be written in place. Scalars arrive as Python
numbers. The return value is ignored.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import load_module
from kernelport.harness.spec import LaunchSpec


def is_available() -> tuple[bool, str]:
    """Can Triton ports run here?"""
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return False, f"triton not importable ({exc}); pip install kernelport[triton]"
    if not torch.cuda.is_available():
        return False, "no CUDA or ROCm GPU visible to torch"
    return True, "ok"


def load_port(path: Path):
    """Import the port and return its run() function."""
    fn = getattr(load_module(path, "kp_triton"), "run", None)
    if fn is None:
        raise AttributeError(f"{Path(path).name} does not define run(...)")
    return fn


def prepare(path: Path, spec: LaunchSpec, data: CaseData):
    """Return (launch_fn, device_tensors)."""
    fn = load_port(path)
    dev = {k: v.to("cuda") for k, v in data.tensors.items()}
    kwargs = {}
    for p in spec.params:
        kwargs[p.name] = dev[p.name] if p.is_tensor else data.scalars[p.name]
    accepted = inspect.signature(fn).parameters
    missing = [k for k in accepted if k not in kwargs and accepted[k].default is inspect.Parameter.empty]
    if missing:
        raise TypeError(f"run() asks for unknown params {missing}; kernel params are {list(kwargs)}")
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    def launch() -> None:
        fn(**kwargs)

    return launch, dev


def run_port(path: Path, spec: LaunchSpec, data: CaseData) -> dict[str, torch.Tensor]:
    """Run once and return output tensors on the CPU."""
    launch, dev = prepare(path, spec, data)
    launch()
    torch.cuda.synchronize()
    return {p.name: dev[p.name].cpu() for p in spec.params if p.is_output}
