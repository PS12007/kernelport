"""Load and run a plain-PyTorch reference.

A reference file defines::

    def reference(a, b, n):          # inputs/inouts and scalars, by name
        return a + b                 # one output, or a dict {name: tensor}

Floating-point inputs are passed as float64 CPU tensors so the reference is
as close to the exact answer as possible. Integer inputs keep their dtype.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import torch

from kernelport.harness.inputs import CaseData
from kernelport.harness.spec import LaunchSpec


def file_hash(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path: Path, prefix: str = "kp_mod") -> ModuleType:
    """Import a Python file under a unique module name."""
    path = Path(path).resolve()
    name = f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}_{file_hash(path)[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # some libraries (triton) need to find the module again
    spec.loader.exec_module(mod)
    return mod


def run_reference(path: Path, spec: LaunchSpec, data: CaseData) -> dict[str, torch.Tensor]:
    """Run the reference for one case and return {output_name: float64/int CPU tensor}."""
    fn = getattr(load_module(path, "kp_ref"), "reference", None)
    if fn is None:
        raise AttributeError(f"{path} does not define reference(...)")
    kwargs = {}
    for p in spec.params:
        if p.kind in ("input", "inout"):
            t = data.tensors[p.name].clone()
            kwargs[p.name] = t.double() if t.is_floating_point() else t
        elif p.kind == "scalar":
            kwargs[p.name] = data.scalars[p.name]
    accepted = inspect.signature(fn).parameters
    if not any(v.kind == v.VAR_KEYWORD for v in accepted.values()):
        missing = [k for k in accepted if k not in kwargs]
        if missing:
            raise TypeError(f"reference() asks for unknown params {missing}; available: {list(kwargs)}")
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    with torch.no_grad():
        out = fn(**kwargs)
    outs = [p for p in spec.params if p.is_output]
    if isinstance(out, torch.Tensor):
        if len(outs) != 1:
            raise TypeError(f"reference returned one tensor but kernel has outputs {[p.name for p in outs]}")
        out = {outs[0].name: out}
    if not isinstance(out, dict):
        raise TypeError("reference must return a tensor or a dict of tensors")
    missing = [p.name for p in outs if p.name not in out]
    if missing:
        raise TypeError(f"reference did not return outputs {missing}")
    return {k: v.detach().cpu() for k, v in out.items()}
