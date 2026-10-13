"""Launch spec: how to call a kernel and which input cases to test.

A spec is plain JSON (``launch.json``) so any machine can regenerate the exact
same inputs. It does not import torch, so the core package can load it.

Example::

    {
      "name": "vector_add",
      "source": "kernel.cu",
      "entry": "vector_add<{T}>",
      "dtypes": ["float32", "float16"],
      "params": [
        {"name": "a",   "kind": "input",  "dtype": "T", "shape": ["n"]},
        {"name": "out", "kind": "output", "dtype": "T", "shape": ["n"]},
        {"name": "n",   "kind": "scalar", "dtype": "int32", "value": "n"}
      ],
      "launch": {"grid": ["cdiv(n, 256)"], "block": [256], "shared_mem": 0},
      "cases": [{"name": "tiny", "dims": {"n": 5}}],
      "init": {"a": {"dist": "normal", "edges": true}}
    }

``dtype: "T"`` is replaced by each entry of ``dtypes`` in turn.
``kind`` is one of input, output, inout, scalar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kernelport.harness.expr import evaluate, evaluate_int

FLOAT_DTYPES = ("float32", "float16", "bfloat16", "float64")
INT_DTYPES = ("int8", "uint8", "int16", "int32", "int64")
ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES

# CUDA C type names used to fill template entries like "kernel<{T}>".
CUDA_TYPE_NAMES = {
    "float32": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "float64": "double",
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "int32": "int",
    "int64": "long long",
}

PARAM_KINDS = ("input", "output", "inout", "scalar")


@dataclass
class ParamSpec:
    """One kernel parameter, in signature order."""

    name: str
    kind: str
    dtype: str
    shape: list[Any] = field(default_factory=list)  # tensors only
    value: Any = None  # scalars only: expression over dims

    def resolved_dtype(self, t: str) -> str:
        """Dtype with the template placeholder ``T`` replaced."""
        return t if self.dtype == "T" else self.dtype

    def resolved_shape(self, dims: dict[str, Any]) -> tuple[int, ...]:
        """Concrete shape for a case."""
        return tuple(evaluate_int(s, dims) for s in self.shape)

    def resolved_value(self, dims: dict[str, Any]) -> Any:
        """Concrete scalar value for a case."""
        return evaluate(self.value, dims)

    @property
    def is_tensor(self) -> bool:
        return self.kind != "scalar"

    @property
    def is_output(self) -> bool:
        return self.kind in ("output", "inout")


@dataclass
class CaseSpec:
    """A named input case: concrete dims plus optional per-case overrides."""

    name: str
    dims: dict[str, Any]
    dtypes: list[str] | None = None  # restrict this case to some dtypes
    init: dict[str, dict] = field(default_factory=dict)  # per-param init overrides


@dataclass
class LaunchSpec:
    """Everything needed to run the original kernel and regenerate its inputs."""

    name: str
    entry: str
    params: list[ParamSpec]
    cases: list[CaseSpec]
    dtypes: list[str] = field(default_factory=lambda: ["float32"])
    source: str = "kernel.cu"
    launch: dict[str, Any] = field(default_factory=dict)
    init: dict[str, dict] = field(default_factory=dict)
    seed: int = 0
    bench_case: str | None = None

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, d: dict) -> "LaunchSpec":
        """Build and validate a spec from parsed JSON."""
        d = dict(d)
        d["params"] = [ParamSpec(**p) for p in d["params"]]
        d["cases"] = [CaseSpec(**c) for c in d["cases"]]
        spec = cls(**d)
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: Path) -> "LaunchSpec":
        """Load a spec from a launch.json file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return asdict(self)

    def hash(self) -> str:
        """Stable hash of the spec contents."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    # ----------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise ValueError on any inconsistency."""
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {names}")
        for t in self.dtypes:
            if t not in ALL_DTYPES:
                raise ValueError(f"unknown dtype {t!r}")
        for p in self.params:
            if p.kind not in PARAM_KINDS:
                raise ValueError(f"param {p.name}: kind must be one of {PARAM_KINDS}")
            if p.dtype != "T" and p.dtype not in ALL_DTYPES:
                raise ValueError(f"param {p.name}: unknown dtype {p.dtype!r}")
            if p.kind == "scalar" and p.value is None:
                raise ValueError(f"scalar param {p.name} needs a value expression")
        if not any(p.is_output for p in self.params):
            raise ValueError("spec has no output or inout parameter")
        for name in self.init:
            if name not in names:
                raise ValueError(f"init given for unknown param {name!r}")
        if not self.cases:
            raise ValueError("spec has no cases")
        for c in self.cases:
            for p in self.params:
                if p.is_tensor:
                    p.resolved_shape(c.dims)
                else:
                    p.resolved_value(c.dims)
        if self.bench_case and self.bench_case not in [c.name for c in self.cases]:
            raise ValueError(f"bench_case {self.bench_case!r} is not a case name")

    # ------------------------------------------------------------- helpers
    def entry_for(self, t: str) -> str:
        """Kernel entry name (template filled) for dtype ``t``."""
        return self.entry.replace("{T}", CUDA_TYPE_NAMES[t])

    def case_dtypes(self, case: CaseSpec) -> list[str]:
        """Dtypes to run for one case."""
        return [t for t in self.dtypes if case.dtypes is None or t in case.dtypes]

    def iter_cases(self):
        """Yield (case, dtype) pairs in a fixed order."""
        for case in self.cases:
            for t in self.case_dtypes(case):
                yield case, t

    def get_case(self, name: str) -> CaseSpec:
        """Find a case by name."""
        for c in self.cases:
            if c.name == name:
                return c
        raise KeyError(name)

    def grid_block(self, dims: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Concrete (grid, block, shared_mem) for the original CUDA launch."""
        grid = tuple(evaluate_int(g, dims) for g in self.launch.get("grid", [1]))
        block = tuple(evaluate_int(b, dims) for b in self.launch.get("block", [1]))
        shmem = evaluate_int(self.launch.get("shared_mem", 0), dims)
        pad = lambda t: t + (1,) * (3 - len(t))  # noqa: E731
        return pad(grid), pad(block), shmem
