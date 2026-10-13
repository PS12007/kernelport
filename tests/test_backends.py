"""Backend registry, capability checks, configs, and core-without-torch packaging."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, ZOO, needs_triton
from kernelport.backends.base import BACKEND_NAMES, get_backend, registered_backends


def test_all_backends_registered():
    assert registered_backends() == list(BACKEND_NAMES)


def test_unknown_backend():
    with pytest.raises(KeyError):
        get_backend("cobol")


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_capability_never_raises(name):
    ok, reason = get_backend(name).is_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason
    assert isinstance(get_backend(name).hardware_name(), str)


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_config_is_sane(name):
    cfg = get_backend(name).config
    assert cfg.name == name
    assert cfg.extension.startswith(".")
    assert "float32" in cfg.dtypes
    for dtype, tol in cfg.tolerances.items():
        assert tol.reason, f"{name}: tolerance override for {dtype} has no documented reason"
    if not cfg.timing_comparable:
        assert cfg.timing_note, f"{name}: non-comparable timings need a note"


def test_core_imports_without_torch():
    """Core package (CLI, analyzer, spec, registry, configs) must not import torch or GPU packages."""
    code = (
        "import sys\n"
        "import kernelport.cli, kernelport.analyzer.cuda_parser, kernelport.analyzer.suggest\n"
        "import kernelport.harness.spec, kernelport.harness.results, kernelport.harness.tolerances\n"
        "from kernelport.backends.base import get_backend, BACKEND_NAMES\n"
        "[get_backend(n).config for n in BACKEND_NAMES]\n"
        "bad = [m for m in ('torch', 'triton', 'cupy', 'numba', 'mlx', 'playwright') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "", f"core imported heavy modules: {out.stdout.strip()}"


@needs_triton
def test_triton_verify_and_bench_record_hardware():
    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import bench_port, verify_port

    kdir = ZOO / "relu"
    spec = LaunchSpec.load(kdir / "launch.json")
    r = verify_port(spec, kdir / "reference.py", kdir / "ports" / "triton.py", "triton")
    assert r.status == "PASS"
    assert r.hardware == get_backend("triton").hardware_name() != "unknown"
    b = bench_port(spec, kdir / spec.source, kdir / "ports" / "triton.py", "triton")
    assert b["port_ms"] > 0 and b["method"] == "cuda_event" and b["hardware"]


@needs_triton
def test_unsupported_dtypes_are_skipped_and_named(monkeypatch):
    from dataclasses import replace

    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import verify_port

    be = get_backend("triton")
    monkeypatch.setattr(be, "config", replace(be.config, dtypes=("float32",)))
    kdir = ZOO / "relu"
    spec = LaunchSpec.load(kdir / "launch.json")
    r = verify_port(spec, kdir / "reference.py", kdir / "ports" / "triton.py", "triton")
    assert r.status == "PASS"
    assert {c.dtype for c in r.cases} == {"float32"}
    assert "bfloat16" in r.reason and "float16" in r.reason


@needs_triton
def test_all_dtypes_unsupported_is_skipped_not_pass(monkeypatch):
    from dataclasses import replace

    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import verify_port

    be = get_backend("triton")
    monkeypatch.setattr(be, "config", replace(be.config, dtypes=("float64",)))
    kdir = ZOO / "relu"
    spec = LaunchSpec.load(kdir / "launch.json")
    r = verify_port(spec, kdir / "reference.py", kdir / "ports" / "triton.py", "triton")
    assert r.status == "SKIPPED"


def test_unavailable_backend_gives_unverified(tmp_path):
    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import verify_port

    port = tmp_path / "port.py"
    port.write_text("def run(x, y, n):\n    pass\n")
    spec = LaunchSpec.load(ZOO / "relu" / "launch.json")
    r = verify_port(spec, ZOO / "relu" / "reference.py", port, "metal" if sys.platform != "darwin" else "hip")
    assert r.status == "UNVERIFIED" and r.reason
    assert r.code_hash  # the saved port is identified even when it cannot run here
