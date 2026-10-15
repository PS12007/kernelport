"""Data models produced by the CUDA analyzer. No torch, no GPU imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Param:
    """One kernel parameter as written in the source."""

    name: str
    type: str  # full type text, e.g. "const T* __restrict__"
    base_type: str  # type without qualifiers/pointer, e.g. "T"
    is_pointer: bool
    is_const: bool  # pointer to const (read-only data) or const scalar
    dtype: str | None = None  # kernelport dtype guess ("float32", "T", ...) or None if unknown
    written: bool = False  # pointer param is assigned through in the body
    read: bool = False  # pointer param is read in the body

    @property
    def kind(self) -> str:
        """Suggested launch-spec kind: input, output, inout, or scalar."""
        if not self.is_pointer:
            return "scalar"
        if self.is_const or not self.written:
            return "input"
        return "inout" if self.read else "output"


@dataclass
class LaunchConfig:
    """A launch site ``kernel<<<grid, block, shmem, stream>>>(args)`` found in the repo."""

    file: str
    line: int
    grid: str
    block: str
    shared_mem: str = "0"
    stream: str = ""
    template_args: str = ""
    args: list[str] = field(default_factory=list)


@dataclass
class KernelInfo:
    """A ``__global__`` kernel definition."""

    id: str  # "<relative path>::<name>", with "@<line>" appended if the name repeats in a file
    name: str
    file: str  # path relative to the scan root, forward slashes
    line: int  # 1-based line of the __global__ keyword
    end_line: int
    params: list[Param]
    template_params: list[str] = field(default_factory=list)  # e.g. ["typename T", "int BLOCK = 256"]
    launch_bounds: str = ""
    extern_c: bool = False
    features: list[str] = field(default_factory=list)  # e.g. ["shared_memory", "atomics"]
    source: str = ""  # full kernel text including template line
    launches: list[LaunchConfig] = field(default_factory=list)
    generated_by: str = ""  # macro invocation that defines the kernel, e.g. "DEFINE_SCALE(float, scale_f32)"

    @property
    def is_template(self) -> bool:
        return bool(self.template_params)

    def to_dict(self, include_source: bool = True) -> dict:
        """JSON-serialisable dict (optionally without the source text)."""
        d = asdict(self)
        for p, pd in zip(self.params, d["params"]):
            pd["kind"] = p.kind
        if not include_source:
            d.pop("source")
        return d

    def summary(self) -> str:
        """One line: id, signature, features."""
        sig = ", ".join(f"{p.type} {p.name}".strip() for p in self.params)
        tpl = f"template<{', '.join(self.template_params)}> " if self.template_params else ""
        feats = f"  [{', '.join(self.features)}]" if self.features else ""
        via = f"  (from {self.generated_by})" if self.generated_by else ""
        return f"{self.id} (line {self.line}): {tpl}{self.name}({sig}){feats}{via}"
