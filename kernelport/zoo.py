"""Run the kernel zoo: validate every original, verify stored ports, benchmark passes.

Layout of each zoo entry::

    zoo/<kernel>/kernel.cu       original CUDA
    zoo/<kernel>/launch.json     launch + input spec
    zoo/<kernel>/reference.py    plain PyTorch reference
    zoo/<kernel>/ports/<backend><ext>   stored hand-written ports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kernelport.backends.base import get_backend
from kernelport.harness.results import RunResult
from kernelport.harness.spec import LaunchSpec

BENCH_ROUNDS = 3  # alternating port/original timing rounds in bench_port


@dataclass
class ZooEntry:
    """Outcome for one zoo kernel."""

    kernel: str
    original: RunResult
    ports: dict[str, RunResult] = field(default_factory=dict)
    bench: dict[str, dict] = field(default_factory=dict)  # backend -> timings


def find_zoo(start: Path | None = None) -> Path:
    """Locate the zoo/ directory (cwd first, then the repo checkout)."""
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


def verify_port(spec: LaunchSpec, reference: Path, port: Path, backend: str,
                stop_after_failures: int | None = None) -> RunResult:
    """Verify a port on ``backend`` against the reference. UNVERIFIED if the backend is unavailable."""
    from kernelport.harness.reference import file_hash
    from kernelport.harness.verify import run_cases

    be = get_backend(backend)
    ok, why = be.is_available()
    if not ok:
        return RunResult(kernel=spec.name, backend=backend, status="UNVERIFIED",
                         hardware=be.hardware_name(), reason=why, code_hash=file_hash(port))
    return run_cases(kernel=spec.name, backend=backend, hardware=be.hardware_name(), spec=spec,
                     reference_path=reference, runner=lambda d: be.run(port, spec, d),
                     code_hash=file_hash(port), tol_overrides=be.config.tolerances,
                     supported_dtypes=be.dtypes(), stop_after_failures=stop_after_failures)


def verify_zoo_port(kdir: Path, port: Path, backend: str = "triton") -> RunResult:
    """verify_port for a zoo directory."""
    return verify_port(LaunchSpec.load(kdir / "launch.json"), kdir / "reference.py", port, backend)


def time_port(spec: LaunchSpec, port: Path, backend: str, dtype: str = "float32", data=None,
              warmup: int | None = None):
    """(Timing, case) of the port alone on the spec's bench case.

    Pass ``data`` (a make_case result) to reuse one input set across many calls, as tuning does:
    regenerating a large case takes long enough on the CPU for the GPU to clock down.
    """
    from kernelport.bench.timer import DEFAULT_WARMUP, time_cuda, time_wall
    from kernelport.harness.inputs import make_case

    be = get_backend(backend)
    case = spec.get_case(spec.bench_case) if spec.bench_case else spec.cases[-1]
    prepared = be.prepare(port, spec, data if data is not None else make_case(spec, case, dtype))
    warmup = DEFAULT_WARMUP if warmup is None else warmup
    if prepared.timer == "cuda_event":
        return time_cuda(prepared.launch, warmup=warmup), case
    return time_wall(prepared.launch, prepared.sync, warmup=warmup), case


def bench_port(spec: LaunchSpec, source: Path, port: Path, backend: str, dtype: str = "float32") -> dict:
    """Time the original CUDA kernel and the port on the spec's bench case.

    The original is timed only when CUDA is available here; otherwise only the port. Port and
    original are timed in BENCH_ROUNDS alternating rounds on one input set, so clock changes hit
    both alike; each reported time is the median of the round medians, with the min-max range.
    """
    import statistics

    from kernelport.bench.timer import time_cuda
    from kernelport.harness import run_cuda
    from kernelport.harness.inputs import make_case

    be = get_backend(backend)
    case = spec.get_case(spec.bench_case) if spec.bench_case else spec.cases[-1]
    data = make_case(spec, case, dtype)
    orig_launch = run_cuda.prepare(source, spec, data)[0] if run_cuda.is_available()[0] else None
    port_rounds, orig_rounds, method = [], [], None
    for _ in range(BENCH_ROUNDS):
        t_port, _ = time_port(spec, port, backend, dtype, data=data)
        port_rounds.append(t_port.median_ms)
        method = t_port.method
        if orig_launch is not None:
            orig_rounds.append(time_cuda(orig_launch).median_ms)
    port_ms = statistics.median(port_rounds)
    out = {"case": case.name, "dtype": dtype, "port_ms": port_ms, "method": method,
           "port_ms_range": [min(port_rounds), max(port_rounds)], "rounds": BENCH_ROUNDS,
           "hardware": be.hardware_name(), "comparable": be.config.timing_comparable,
           "note": be.config.timing_note, "original_ms": None, "speedup": None}
    if orig_rounds:
        out["original_ms"] = statistics.median(orig_rounds)
        out["original_ms_range"] = [min(orig_rounds), max(orig_rounds)]
        out["speedup"] = out["original_ms"] / port_ms
    spreads = [hi / lo for lo, hi in (out["port_ms_range"], out.get("original_ms_range", [1, 1])) if lo > 0]
    out["noisy"] = any(s > 1.2 for s in spreads)
    return out



def run_zoo(zoo: Path, kernels: list[str] | None = None, backends: tuple[str, ...] = ("triton",),
            bench: bool = True, out_dir: Path | None = None) -> list[ZooEntry]:
    """Validate originals, verify stored ports, and benchmark the passing ones."""
    from kernelport.hardware import gpu_name

    hw = gpu_name() or "no GPU"
    entries = []
    for name in kernels or list_kernels(zoo):
        kdir = zoo / name
        spec = LaunchSpec.load(kdir / "launch.json")
        entry = ZooEntry(name, validate_original(kdir, hw))
        for backend in backends:
            be = get_backend(backend)
            port = kdir / "ports" / f"{backend}{be.config.extension}"
            if not port.is_file():
                entry.ports[backend] = RunResult(kernel=name, backend=backend, status="SKIPPED",
                                                 hardware=be.hardware_name(), reason="no stored port")
                continue
            if entry.original.status == "FAIL":
                entry.ports[backend] = RunResult(kernel=name, backend=backend, status="SKIPPED",
                                                 hardware=be.hardware_name(),
                                                 reason="original did not pass its reference")
                continue
            res = verify_port(spec, kdir / "reference.py", port, backend)
            entry.ports[backend] = res
            if bench and res.status == "PASS":
                entry.bench[backend] = bench_port(spec, kdir / spec.source, port, backend)
        if out_dir:
            entry.original.save(out_dir / name / "cuda" / "result.json")
            for b, r in entry.ports.items():
                r.save(out_dir / name / b / "result.json")
        entries.append(entry)
    return entries
