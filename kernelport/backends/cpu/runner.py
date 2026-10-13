"""Run a cpu (Numba) port.

A cpu port is a Python file defining::

    def run(<same parameter names as launch.json>):
        ...call one or more @numba.njit functions...

Tensor parameters arrive as C-contiguous NumPy arrays with the launch.json
shapes (outputs pre-filled with NaN, or a sentinel for ints) and must be
written in place. Scalars arrive as Python numbers. run() must return None.

Numba compiles on the first call for each argument type, so the loaded run()
is cached per file content: verification compiles once per dtype, not once
per case, and benchmark warm-up excludes compilation.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kernelport.backends.base import Prepared, bind_kwargs
from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import file_hash, load_module
from kernelport.harness.spec import LaunchSpec

_loaded: dict[tuple[str, str], object] = {}


def load_port(path: Path):
    """Import the port (once per content hash) and return its run() function."""
    path = Path(path).resolve()
    key = (str(path), file_hash(path))
    if key not in _loaded:
        fn = getattr(load_module(path, "kp_cpu"), "run", None)
        if fn is None:
            raise AttributeError(f"{path.name} does not define run(...)")
        _loaded[key] = fn
    return _loaded[key]


def _to_numpy(name: str, t: torch.Tensor):
    if t.dtype == torch.bfloat16:
        raise TypeError(f"{name}: bfloat16 has no NumPy dtype; the cpu backend does not run bfloat16 cases")
    return t.clone().contiguous().numpy()


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Load the port and bind it to one case's arrays."""
    fn = load_port(path)
    arrays = {k: _to_numpy(k, v) for k, v in data.tensors.items()}
    kwargs = bind_kwargs(fn, spec, arrays, data.scalars)
    outputs = [p.name for p in spec.params if p.is_output]

    def launch() -> None:
        ret = fn(**kwargs)
        if ret is not None:
            raise TypeError("run() returned a value; outputs must be written in place "
                            "(e.g. y[:] = ... or y[i] = ...), and run() must return None")

    return Prepared(
        launch=launch,
        sync=lambda: None,
        fetch=lambda: {n: torch.from_numpy(arrays[n].copy()) for n in outputs},
        timer="wall_clock",
    )
