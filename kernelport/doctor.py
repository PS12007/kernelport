"""Environment report: OS, Python, torch, GPU, installed extras, and a smoke
kernel for every backend that is available on this machine.

Every check is isolated: a missing package or a failing backend is reported
with a reason, never raised.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable

# Package that marks each optional extra as installed.
EXTRA_MARKERS: dict[str, list[str]] = {
    "ref": ["torch"],
    "cuda": ["torch", "cupy"],
    "triton": ["torch", "triton"],
    "metal": ["mlx"],
    "webgpu": ["playwright"],
    "cpu": ["numba"],
    "agent": ["anthropic"],
}


@dataclass
class BackendCheck:
    """Result of checking one backend on this machine."""

    name: str
    available: bool
    reason: str
    smoke: str = "skipped"  # "pass", "fail", or "skipped"
    detail: str = ""


@dataclass
class DoctorReport:
    """Full environment report."""

    os: str
    python: str
    torch: str
    cuda_available: bool
    gpu_name: str
    compute_capability: str
    extras: dict[str, bool] = field(default_factory=dict)
    backends: list[BackendCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @property
    def ok(self) -> bool:
        """True when no available backend failed its smoke test."""
        return all(b.smoke != "fail" for b in self.backends)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _torch_info() -> tuple[str, bool, str, str]:
    if not _has("torch"):
        return "not installed", False, "", ""
    try:
        import torch

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            return torch.__version__, True, torch.cuda.get_device_name(0), f"sm_{cc[0]}{cc[1]}"
        return torch.__version__, False, "", ""
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}", False, "", ""


# ---------------------------------------------------------------- smoke tests
# Each returns None on success or raises. Inputs are tiny and fixed.

def _smoke_cuda() -> None:
    import cupy as cp
    import numpy as np

    src = r"""extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }"""
    kern = cp.RawKernel(src, "add")
    n = 1000
    a = cp.arange(n, dtype=cp.float32)
    b = cp.full(n, 2.0, dtype=cp.float32)
    c = cp.empty_like(a)
    kern(((n + 255) // 256,), (256,), (a, b, c, np.int32(n)))
    if not bool(cp.allclose(c, a + 2.0)):
        raise AssertionError("CuPy RawKernel produced wrong result")


def _smoke_triton() -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    n = 1000
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full((n,), 2.0, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    if not torch.equal(out, x + 2.0):
        raise AssertionError("Triton kernel produced wrong result")


def _smoke_cpu() -> None:
    import numpy as np
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def add(a, b, c):
        for i in prange(a.shape[0]):
            c[i] = a[i] + b[i]

    a = np.arange(1000, dtype=np.float32)
    c = np.empty_like(a)
    add(a, np.full_like(a, 2.0), c)
    if not np.array_equal(c, a + 2.0):
        raise AssertionError("Numba kernel produced wrong result")


def _smoke_metal() -> None:
    import mlx.core as mx

    kern = mx.fast.metal_kernel(
        name="add",
        input_names=["a"],
        output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = a[i] + 2.0f;",
    )
    a = mx.arange(1000, dtype=mx.float32)
    (out,) = kern(
        inputs=[a], grid=(1000, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[a.shape], output_dtypes=[a.dtype],
    )
    if not bool(mx.array_equal(out, a + 2.0)):
        raise AssertionError("MLX metal kernel produced wrong result")


# --------------------------------------------------------------- capabilities

def _cuda_gpu() -> tuple[bool, str]:
    if not _has("torch"):
        return False, "torch not installed (pip install kernelport[cuda])"
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (CPU-only torch or no NVIDIA GPU)"
    if getattr(torch.version, "hip", None):
        return False, "ROCm GPU, not NVIDIA CUDA"
    return True, "ok"


def _check_cuda() -> tuple[bool, str]:
    ok, why = _cuda_gpu()
    if not ok:
        return ok, why
    if not _has("cupy"):
        return False, "cupy not installed (pip install kernelport[cuda])"
    return True, "CuPy + NVRTC on NVIDIA GPU"


def _check_triton() -> tuple[bool, str]:
    if not _has("triton"):
        return False, "triton not installed (pip install kernelport[triton])"
    if not _has("torch"):
        return False, "torch not installed"
    import torch

    if not torch.cuda.is_available():
        return False, "no CUDA or ROCm GPU visible to torch"
    return True, "Triton on " + ("ROCm" if getattr(torch.version, "hip", None) else "CUDA")


def _check_cpu() -> tuple[bool, str]:
    if not _has("numba"):
        return False, "numba not installed (pip install kernelport[cpu])"
    return True, "Numba on CPU"


def _check_metal() -> tuple[bool, str]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False, "Metal needs macOS on Apple Silicon"
    if not _has("mlx"):
        return False, "mlx not installed (pip install kernelport[metal])"
    return True, "MLX metal_kernel on Apple GPU"


def _check_webgpu() -> tuple[bool, str]:
    if not _has("playwright"):
        return False, "playwright not installed (pip install kernelport[webgpu]); added in M12"
    return False, "WebGPU runner not implemented yet (M12)"


def _check_hip() -> tuple[bool, str]:
    if not _has("torch"):
        return False, "torch not installed"
    import torch

    if not getattr(torch.version, "hip", None):
        return False, "needs a ROCm build of torch on an AMD GPU"
    return False, "HIP runner not implemented yet (M14)"


BACKENDS: list[tuple[str, Callable[[], tuple[bool, str]], Callable[[], None] | None]] = [
    ("cuda (original)", _check_cuda, _smoke_cuda),
    ("triton", _check_triton, _smoke_triton),
    ("cpu", _check_cpu, _smoke_cpu),
    ("metal", _check_metal, _smoke_metal),
    ("webgpu", _check_webgpu, None),
    ("hip", _check_hip, None),
]


def _run_backend(name: str, check: Callable, smoke: Callable | None) -> BackendCheck:
    try:
        ok, reason = check()
    except Exception as exc:
        return BackendCheck(name, False, f"capability check crashed: {exc}")
    result = BackendCheck(name, ok, reason)
    if ok and smoke is not None:
        try:
            smoke()
            result.smoke = "pass"
        except Exception as exc:
            result.smoke = "fail"
            result.detail = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3)
    return result


def run_doctor() -> DoctorReport:
    """Collect environment info and smoke-test every available backend."""
    torch_ver, cuda_ok, gpu, cc = _torch_info()
    report = DoctorReport(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=sys.version.split()[0],
        torch=torch_ver,
        cuda_available=cuda_ok,
        gpu_name=gpu,
        compute_capability=cc,
        extras={k: all(_has(m) for m in mods) for k, mods in EXTRA_MARKERS.items()},
    )
    report.backends = [_run_backend(*b) for b in BACKENDS]
    return report


def print_report(report: DoctorReport) -> None:
    """Pretty-print a doctor report with rich."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    con = Console()
    con.print("[bold]kernelport doctor[/bold]")
    con.print(f"OS:       {report.os}")
    con.print(f"Python:   {report.python}")
    con.print(f"torch:    {report.torch}")
    con.print(f"CUDA:     {report.cuda_available}")
    if report.gpu_name:
        con.print(f"GPU:      {report.gpu_name} ({report.compute_capability})")
    con.print("Extras:   " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in report.extras.items()))

    table = Table(title="Backends")
    for col in ("backend", "available", "smoke", "reason"):
        table.add_column(col)
    colors = {"pass": "green", "fail": "red", "skipped": "dim"}
    for b in report.backends:
        table.add_row(b.name, "yes" if b.available else "no",
                      f"[{colors[b.smoke]}]{b.smoke}[/]", escape(b.reason))
    con.print(table)
    for b in report.backends:
        if b.smoke == "fail":
            con.print(f"[red]{b.name} smoke failed:[/red]\n{b.detail}")
    con.print("[green]doctor: OK[/green]" if report.ok else "[red]doctor: FAILED[/red]")
