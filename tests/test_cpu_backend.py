"""cpu (Numba) backend: every zoo port passes, broken ports fail for the right reason. No GPU needed."""

from pathlib import Path

import pytest

from conftest import ZOO, _has
from kernelport.backends.base import get_backend

needs_numba = pytest.mark.skipif(not (_has("numba") and _has("torch")), reason="needs numba + torch")
BAD = Path(__file__).parent / "bad_ports" / "cpu"
KERNELS = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/cpu.py"))


def _verify(kernel, port):
    from kernelport.zoo import verify_zoo_port

    return verify_zoo_port(ZOO / kernel, port, backend="cpu")


def test_every_zoo_kernel_has_a_cpu_port():
    assert len(KERNELS) == 12


@needs_numba
def test_cpu_available_with_named_hardware():
    ok, reason = get_backend("cpu").is_available()
    assert ok, reason
    assert "Numba" in reason
    assert "threads" in get_backend("cpu").hardware_name()


@needs_numba
@pytest.mark.parametrize("kernel", KERNELS)
def test_zoo_cpu_port_passes(kernel):
    r = _verify(kernel, ZOO / kernel / "ports" / "cpu.py")
    assert r.status == "PASS", r.failure_report()
    assert r.hardware == get_backend("cpu").hardware_name()
    assert {c.dtype for c in r.cases} == {"float32"}
    assert "not tested: " in r.reason and "float16" in r.reason  # skipped dtypes are named


@needs_numba
@pytest.mark.parametrize("kernel,port,expect", [
    ("vector_add", "vector_add_missing_tail.py", "NaN"),
    ("relu", "relu_returns_array.py", "returned a value"),
    ("relu", "relu_rebinds_output.py", "NaN"),
    ("cumsum", "cumsum_exclusive.py", "outside tolerance"),
    ("reduce_sum", "reduce_sum_overwrites.py", "outside tolerance"),
    ("layernorm", "layernorm_unbiased.py", "outside tolerance"),
])
def test_bad_cpu_port_fails(kernel, port, expect):
    r = _verify(kernel, BAD / port)
    assert r.status == "FAIL"
    assert expect in r.failure_report(), r.failure_report()


@needs_numba
def test_overwrite_bug_only_caught_by_accumulate_case():
    r = _verify("reduce_sum", BAD / "reduce_sum_overwrites.py")
    failed = {c.case for c in r.cases if not c.passed}
    assert failed == {"accumulate_1000"}


@needs_numba
def test_cpu_bench_is_wall_clock_and_not_comparable():
    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import bench_port

    kdir = ZOO / "relu"
    spec = LaunchSpec.load(kdir / "launch.json")
    spec.bench_case = "odd_4097"  # keep the test fast
    b = bench_port(spec, kdir / spec.source, kdir / "ports" / "cpu.py", "cpu")
    assert b["method"] == "wall_clock" and b["port_ms"] > 0
    assert b["comparable"] is False and b["note"]


@needs_numba
def test_compiled_once_per_file_content(tmp_path):
    """Verification reuses the loaded port across cases; a changed file is reloaded."""
    from kernelport.backends.cpu.runner import load_port

    port = tmp_path / "p.py"
    port.write_text("def run(x, y, n):\n    y[:] = x\n")
    assert load_port(port) is load_port(port)
    first = load_port(port)
    port.write_text("def run(x, y, n):\n    y[:] = 2 * x\n")
    assert load_port(port) is not first


@needs_numba
def test_tool_workflow_on_cpu_without_gpu(tmp_path):
    """write_port -> verify_port -> bench_port -> report through the tools, reference frozen synthetically."""
    import json
    import shutil

    from kernelport import tools
    from kernelport.analyzer.cuda_parser import scan
    from kernelport.harness.results import CaseResult, RunResult
    from kernelport.workspace import Workspace, sha256_file

    repo = tmp_path / "repo"
    shutil.copytree(ZOO / "relu", repo / "relu", ignore=shutil.ignore_patterns("ports", "__pycache__"))
    ws = Workspace(repo)
    for k in scan(repo):
        ws.register(k)
    ws.write_spec("relu", json.loads((ZOO / "relu" / "launch.json").read_text()))
    ws.write_reference("relu", (ZOO / "relu" / "reference.py").read_text())
    ws.freeze("relu", RunResult(kernel="relu", backend="cuda", status="PASS", hardware="elsewhere",
                                reference_hash=sha256_file(ws.get("relu").dir / "reference.py"),
                                code_hash="src", cases=[CaseResult("c", "float32", True, "h")]))

    bad = (BAD / "relu_rebinds_output.py").read_text()
    assert tools.write_port("relu", "cpu", bad, root=repo)["backend_available"]
    v = tools.verify_port("relu", "cpu", root=repo)
    assert v["status"] == "FAIL" and v["attempts_remaining"] == 5
    assert not tools.bench_port("relu", "cpu", root=repo)["ok"]

    tools.write_port("relu", "cpu", (ZOO / "relu" / "ports" / "cpu.py").read_text(), root=repo)
    v = tools.verify_port("relu", "cpu", root=repo)
    assert v["status"] == "PASS" and v["attempt"] == 2, v
    b = tools.bench_port("relu", "cpu", {"case": "odd_4097"}, root=repo)
    assert b["ok"] and "Not comparable" in b["summary"]

    md = tools.make_report("markdown", root=repo)["content"]
    assert "| `relu` | frozen | PASS (" in md  # PASS (<this machine>)
    assert "(not comparable)" in md
