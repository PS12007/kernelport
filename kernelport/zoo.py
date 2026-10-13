"""Run the kernel zoo: validate every original, verify stored ports, benchmark passes.

Layout of each zoo entry::

    zoo/<kernel>/kernel.cu       original CUDA
    zoo/<kernel>/launch.json     launch + input spec
    zoo/<kernel>/reference.py    plain PyTorch reference
    zoo/<kernel>/ports/<backend>.py   stored hand-written ports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kernelport.harness.results import RunResult
from kernelport.harness.spec import LaunchSpec

PORT_BACKENDS = ("triton",)


@dataclass
class ZooEntry:
    """Outcome for one zoo kernel."""

    kernel: str
    original: RunResult
    ports: dict[str, RunResult] = field(default_factory=dict)
    bench: dict[str, dict] = field(default_factory=dict)  # backend -> timings


def find_zoo(start: Path | None = None) -> Path:
    """Locate the zoo/ directory (repo checkout or cwd)."""
    for base in [start or Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (base / "zoo").is_dir():
            return base / "zoo"
    raise FileNotFoundError("no zoo/ directory found")


def list_kernels(zoo: Path) -> list[str]:
    """Zoo kernel names, sorted."""
    return sorted(p.name for p in zoo.iterdir() if (p / "launch.json").is_file())


def validate_original(kdir: Path, hardware: str, reference: Path | None = None,
                      stop_after_failures: int | None = None) -> RunResult:
    """Run the original CUDA kernel against its reference (or an alternative reference file)."""
    from kernelport.harness import run_cuda
    from kernelport.harness.verify import run_cases

    spec = LaunchSpec.load(kdir / "launch.json")
    ok, why = run_cuda.is_available()
    if not ok:
        return RunResult(kernel=spec.name, backend="cuda", status="UNVERIFIED", hardware=hardware, reason=why)
    src = kdir / spec.source
    return run_cases(kernel=spec.name, backend="cuda", hardware=hardware, spec=spec,
                     reference_path=reference or kdir / "reference.py",
                     runner=lambda d: run_cuda.run_original(src, spec, d),
                     stop_after_failures=stop_after_failures,
                     code_hash=run_cuda.source_hash(src))


def verify_triton(kdir: Path, port: Path, hardware: str) -> RunResult:
    """Verify a Triton port against the reference."""
    from kernelport.harness import run_triton
    from kernelport.harness.reference import file_hash
    from kernelport.harness.verify import run_cases

    spec = LaunchSpec.load(kdir / "launch.json")
    ok, why = run_triton.is_available()
    if not ok:
        return RunResult(kernel=spec.name, backend="triton", status="UNVERIFIED", hardware=hardware, reason=why)
    return run_cases(kernel=spec.name, backend="triton", hardware=hardware, spec=spec,
                     reference_path=kdir / "reference.py",
                     runner=lambda d: run_triton.run_port(port, spec, d),
                     code_hash=file_hash(port))


def bench_pair(kdir: Path, port: Path, dtype: str = "float32") -> dict:
    """Time original vs Triton port on the spec's bench case."""
    from kernelport.bench.timer import time_cuda
    from kernelport.harness import run_cuda, run_triton
    from kernelport.harness.inputs import make_case

    spec = LaunchSpec.load(kdir / "launch.json")
    case = spec.get_case(spec.bench_case) if spec.bench_case else spec.cases[-1]
    data = make_case(spec, case, dtype)
    orig_launch, _ = run_cuda.prepare(kdir / spec.source, spec, data)
    port_launch, _ = run_triton.prepare(port, spec, data)
    t_orig = time_cuda(orig_launch)
    t_port = time_cuda(port_launch)
    return {"case": case.name, "dtype": dtype, "original_ms": t_orig.median_ms,
            "port_ms": t_port.median_ms, "speedup": t_orig.median_ms / t_port.median_ms,
            "method": t_port.method}


def run_zoo(zoo: Path, kernels: list[str] | None = None, backends: tuple[str, ...] = PORT_BACKENDS,
            bench: bool = True, out_dir: Path | None = None) -> list[ZooEntry]:
    """Validate originals, verify stored ports, and benchmark the passing ones."""
    from kernelport.hardware import gpu_name

    hw = gpu_name() or "no GPU"
    entries = []
    for name in kernels or list_kernels(zoo):
        kdir = zoo / name
        entry = ZooEntry(name, validate_original(kdir, hw))
        for backend in backends:
            port = kdir / "ports" / f"{backend}.py"
            if not port.is_file():
                entry.ports[backend] = RunResult(kernel=name, backend=backend, status="SKIPPED",
                                                 hardware=hw, reason="no stored port")
                continue
            if entry.original.status != "PASS":
                entry.ports[backend] = RunResult(kernel=name, backend=backend, status="SKIPPED",
                                                 hardware=hw, reason="original did not pass its reference")
                continue
            res = verify_triton(kdir, port, hw)
            entry.ports[backend] = res
            if bench and res.status == "PASS":
                entry.bench[backend] = bench_pair(kdir, port)
        if out_dir:
            entry.original.save(out_dir / name / "cuda" / "result.json")
            for b, r in entry.ports.items():
                r.save(out_dir / name / b / "result.json")
        entries.append(entry)
    return entries
