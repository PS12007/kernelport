"""Backend interface and registry.

A backend lives in ``kernelport/backends/<name>/`` with three modules:

- ``config.py``     CONFIG: BackendConfig (no heavy imports)
- ``capability.py`` is_available() -> (bool, reason) and hardware_name() -> str
                    (must import cleanly even when the backend's packages are missing)
- ``runner.py``     prepare(port_path, spec, data) -> Prepared  (imports heavy packages lazily)

The registry imports config and capability eagerly-on-demand and the runner
only when something is actually run, so ``kernelport`` works on machines that
have none of the GPU packages.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kernelport.harness.tolerances import Tolerance

BACKEND_NAMES = ("triton", "cpu", "metal", "webgpu", "hip")


@dataclass(frozen=True)
class BackendConfig:
    """Static description of a backend."""

    name: str
    extension: str  # port file extension, e.g. ".py"
    description: str
    dtypes: tuple[str, ...]  # output/compute dtypes this backend can run
    tolerances: dict[str, Tolerance] = field(default_factory=dict)  # overrides, each with a reason
    timing_comparable: bool = True  # False for CPU / browser numbers
    timing_note: str = ""
    hardware_hint: str = ""  # where verification can happen, e.g. "Apple Silicon Mac"
    port_template: str = ""  # skeleton shown to agents
    tune_space: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class Prepared:
    """A port loaded and bound to one case's inputs, ready to launch repeatedly."""

    launch: Callable[[], None]  # run once (may be async on GPUs)
    sync: Callable[[], None]  # wait for completion
    fetch: Callable[[], dict[str, Any]]  # output tensors as CPU torch tensors
    timer: str = "wall_clock"  # "cuda_event" when launch runs on the current torch CUDA stream


class Backend:
    """Facade over a backend's config, capability and runner modules."""

    def __init__(self, name: str) -> None:
        if name not in BACKEND_NAMES:
            raise KeyError(f"unknown backend {name!r}; known: {', '.join(BACKEND_NAMES)}")
        self.name = name
        self._pkg = f"kernelport.backends.{name}"
        self.config: BackendConfig = importlib.import_module(f"{self._pkg}.config").CONFIG
        self._cap = importlib.import_module(f"{self._pkg}.capability")

    def is_available(self) -> tuple[bool, str]:
        """(available, reason). Never raises."""
        try:
            return self._cap.is_available()
        except Exception as exc:  # a broken install must not crash callers
            return False, f"capability check failed: {type(exc).__name__}: {exc}"

    def hardware_name(self) -> str:
        """Name of the device ports run on here."""
        try:
            return self._cap.hardware_name()
        except Exception:
            return "unknown"

    def prepare(self, port: Path, spec, data) -> Prepared:
        """Load ``port`` and bind it to one case."""
        runner = importlib.import_module(f"{self._pkg}.runner")
        return runner.prepare(Path(port), spec, data)

    def run(self, port: Path, spec, data) -> dict[str, Any]:
        """Run once and return outputs as CPU tensors."""
        p = self.prepare(port, spec, data)
        p.launch()
        p.sync()
        return p.fetch()

    def dtypes(self) -> tuple[str, ...]:
        """dtypes runnable here: the config's, narrowed by the capability module if it knows better
        (e.g. float16 only on WebGPU adapters with shader-f16)."""
        fn = getattr(self._cap, "supported_dtypes", None)
        if fn is None:
            return self.config.dtypes
        try:
            return tuple(d for d in self.config.dtypes if d in fn())
        except Exception:
            return self.config.dtypes

    def supports(self, dtype: str) -> bool:
        """Whether this backend runs cases of ``dtype`` here."""
        return dtype in self.dtypes()


def bind_kwargs(fn, spec, tensors: dict, scalars: dict) -> dict:
    """Match launch.json params to run()'s signature, by name."""
    available = {p.name: (tensors[p.name] if p.is_tensor else scalars[p.name]) for p in spec.params}
    accepted = inspect.signature(fn).parameters
    if any(v.kind == v.VAR_KEYWORD for v in accepted.values()):
        return available
    missing = [k for k, v in accepted.items() if k not in available and v.default is inspect.Parameter.empty]
    if missing:
        raise TypeError(f"run() asks for unknown params {missing}; launch.json params are {list(available)}")
    return {k: v for k, v in available.items() if k in accepted}


_cache: dict[str, Backend] = {}


def get_backend(name: str) -> Backend:
    """Backend by name (cached)."""
    if name not in _cache:
        _cache[name] = Backend(name)
    return _cache[name]


def registered_backends() -> list[str]:
    """Names of backends whose modules exist in this install (implemented or not)."""
    names = []
    for n in BACKEND_NAMES:
        try:
            importlib.import_module(f"kernelport.backends.{n}.config")
            names.append(n)
        except ModuleNotFoundError:
            continue
    return names
