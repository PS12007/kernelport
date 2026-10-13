"""Run a Triton port.

A Triton port is a Python file defining::

    def run(<same parameter names as launch.json>):
        ...launch one or more @triton.jit kernels...

Tensor parameters arrive as GPU tensors (outputs pre-filled with NaN, or a
sentinel for ints) and must be written in place. Scalars arrive as Python
numbers. The return value is ignored.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kernelport.backends.base import Prepared, bind_kwargs
from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import load_module
from kernelport.harness.spec import LaunchSpec


def load_port(path: Path):
    """Import the port and return its run() function."""
    fn = getattr(load_module(path, "kp_triton"), "run", None)
    if fn is None:
        raise AttributeError(f"{Path(path).name} does not define run(...)")
    return fn


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Load the port and bind it to one case's tensors on the GPU."""
    fn = load_port(path)
    dev = {k: v.to("cuda") for k, v in data.tensors.items()}
    kwargs = bind_kwargs(fn, spec, dev, data.scalars)
    outputs = [p.name for p in spec.params if p.is_output]
    return Prepared(
        launch=lambda: fn(**kwargs),
        sync=torch.cuda.synchronize,
        fetch=lambda: {n: dev[n].cpu() for n in outputs},
        timer="cuda_event",
    )
