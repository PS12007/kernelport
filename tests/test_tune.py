"""Autotuning: TUNE parsing and variant rendering (no GPU), plus a real tune on CUDA + Triton."""

import json
import shutil

import pytest

from conftest import ZOO, needs_cuda, needs_triton
from kernelport.tune import MAX_VARIANTS, TuneError, read_space, render, variants

PORT = '''import triton

BLOCK = 1024  # rows are chunked by this
NUM_WARPS = (
    4
)
TUNE = {"BLOCK": [256, 1024], "NUM_WARPS": [2, 4, 8]}


def run(x, y, n):
    pass
'''


def test_read_space_and_current():
    space, current = read_space(PORT)
    assert space == {"BLOCK": [256, 1024], "NUM_WARPS": [2, 4, 8]}
    assert current == {"BLOCK": 1024, "NUM_WARPS": 4}


def test_variants_cover_product_and_include_current():
    combos, current = variants(PORT)
    assert len(combos) == 6 and current in combos
    combos, current = variants(PORT.replace("BLOCK = 1024", "BLOCK = 512"))
    assert len(combos) == 7 and combos[0] == {"BLOCK": 512, "NUM_WARPS": 4}


def test_render_rewrites_only_named_assignments():
    out = render(PORT, {"BLOCK": 256, "NUM_WARPS": 8})
    assert "BLOCK = 256  # rows are chunked by this\n" in out
    assert "NUM_WARPS = 8\n" in out and "(\n    4\n)" not in out
    assert read_space(out)[1] == {"BLOCK": 256, "NUM_WARPS": 8}
    assert out.endswith("def run(x, y, n):\n    pass\n")
    crlf = render(PORT.replace("\n", "\r\n"), {"BLOCK": 256})
    assert "BLOCK = 256  # rows are chunked by this\r\n" in crlf and "\r\r" not in crlf


@pytest.mark.parametrize("code,msg", [
    ("BLOCK = 1\n", "no TUNE"),
    ("BLOCK = 1\nTUNE = dict(BLOCK=[1])\n", "literal dict"),
    ("TUNE = {'BLOCK': [1, 2]}\n", "no module-level `BLOCK"),
    ("BLOCK = 1\nTUNE = {'BLOCK': []}\n", "non-empty list"),
    ("BLOCK = 1\nTUNE = {'BLOCK': [[1]]}\n", "numbers, strings or booleans"),
    ("BLOCK = f()\nTUNE = {'BLOCK': [1]}\n", "literal value"),
    (f"A = 1\nB = 1\nTUNE = {{'A': list(range(9)), 'B': list(range({MAX_VARIANTS}))}}\n", "literal dict"),
    (f"A = 1\nB = 1\nTUNE = {{'A': {list(range(9))}, 'B': {list(range(4))}}}\n", "at most"),
    ("def f(:\n", "syntax error"),
])
def test_bad_tune_rejected(code, msg):
    with pytest.raises(TuneError, match=msg):
        read_space(code)


CPU_RELU = '''import numpy as np

CHUNK = 1
OFFSET = 0.0
TUNE = {"CHUNK": [1, 2], "OFFSET": [0.0, 1.0]}


def run(x, y, n):
    np.maximum(x[:n], x.dtype.type(0), out=y[:n])
    if OFFSET:
        y[:n] += OFFSET  # wrong on purpose: those variants must FAIL, never win
'''


@pytest.fixture
def cpu_ws(tmp_path):
    """relu with a frozen reference (fake validation) and a passing TUNE-able cpu port."""
    from kernelport import tools
    from kernelport.harness.results import CaseResult, RunResult
    from kernelport.workspace import Workspace, sha256_file

    pytest.importorskip("numba")
    pytest.importorskip("torch")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    shutil.copy(ZOO / "relu" / "kernel.cu", repo / "relu.cu")
    spec = json.loads((ZOO / "relu" / "launch.json").read_text())
    spec["source"] = "relu.cu"
    root = str(repo)
    tools.scan_kernels(".", root=root)
    tools.write_reference("relu", (ZOO / "relu" / "reference.py").read_text(), spec=spec, root=root)
    ws = Workspace(repo)
    ws.freeze("relu", RunResult(kernel="relu", backend="cuda", status="PASS", hardware="Test GPU",
                                reference_hash=sha256_file(ws.get("relu").dir / "reference.py"),
                                code_hash="src", cases=[CaseResult("tiny", "float32", True, "h")]))
    tools.write_port("relu", "cpu", CPU_RELU, root=root)
    assert tools.verify_port("relu", "cpu", root=root)["status"] == "PASS"
    return ws


def _fake_timer(monkeypatch, ms_for):
    """Replace port timing: ms_for(CHUNK, call_number) -> milliseconds."""
    from kernelport import zoo
    from kernelport.bench.timer import Timing

    calls = []

    def fake(spec, port, backend, dtype="float32"):
        chunk = 2 if "CHUNK = 2" in port.read_text() else 1
        calls.append(chunk)
        ms = ms_for(chunk, len(calls))
        return Timing(ms, ms, ms, 1, "wall_clock"), spec.cases[-1]

    monkeypatch.setattr(zoo, "time_port", fake)
    return calls


def test_tune_keeps_clear_winner_and_reverifies(cpu_ws, monkeypatch):
    from kernelport import tools

    calls = _fake_timer(monkeypatch, lambda chunk, i: 0.5 if chunk == 2 else 1.0)
    res = tools.tune_port("relu", "cpu", root=str(cpu_ws.root))
    assert res["ok"] and res["changed"], res
    assert res["best"] == {"CHUNK": 2, "OFFSET": 0.0} and res["gain"] == pytest.approx(2.0)
    assert (res["variants_passed"], res["variants_total"]) == (2, 4)
    assert sorted(r["status"] for r in res["rows"]) == ["FAIL", "FAIL", "PASS", "PASS"]
    assert calls == [1, 2] * 5                       # interleaved rounds, failing variants never timed
    port = cpu_ws.port_path("relu", "cpu").read_text()
    assert "CHUNK = 2" in port and "OFFSET = 0.0" in port
    (row,) = cpu_ws.status()
    assert row["status"] == "PASS"                   # the new code was verified again


def test_tune_ignores_noisy_win(cpu_ws, monkeypatch):
    from kernelport import tools

    # Median says CHUNK=2 is 20% faster, but one round of it is slower than the current one.
    times = {1: [1.0, 1.0, 1.0, 1.0, 1.0], 2: [0.8, 0.8, 1.1, 0.8, 0.8]}
    _fake_timer(monkeypatch, lambda chunk, i: times[chunk][(i - 1) // 2])
    before = cpu_ws.port_path("relu", "cpu").read_text()
    res = tools.tune_port("relu", "cpu", root=str(cpu_ws.root))
    assert res["ok"] and not res["changed"]
    assert "no passing variant was clearly faster" in res["summary"]
    assert cpu_ws.port_path("relu", "cpu").read_text() == before


def test_tune_refusals(cpu_ws):
    from kernelport import tools

    root = str(cpu_ws.root)
    tools.write_port("relu", "cpu", CPU_RELU.replace("TUNE = ", "XTUNE = "), root=root)
    assert "verify_port first" in tools.tune_port("relu", "cpu", root=root)["summary"]
    tools.verify_port("relu", "cpu", root=root)
    assert "no TUNE" in tools.tune_port("relu", "cpu", root=root)["summary"]
    assert "not supported by cpu" in tools.tune_port("relu", "cpu", dtype="float16", root=root)["summary"]


SOFTMAX = (ZOO / "softmax" / "ports" / "triton.py").read_text()
TUNED_SOFTMAX = SOFTMAX.replace(
    "def run(x, y, rows, cols):\n    BLOCK = min(triton.next_power_of_2(cols), 1024)\n",
    "BLOCK = 128\nTUNE = {\"BLOCK\": [128, 1024]}\n\n\ndef run(x, y, rows, cols):\n",
).replace("BLOCK=BLOCK)", "BLOCK=min(BLOCK, triton.next_power_of_2(cols)))")


@needs_cuda
@needs_triton
def test_tune_port_end_to_end(tmp_path):
    from kernelport import tools
    from kernelport.workspace import Workspace

    assert TUNED_SOFTMAX != SOFTMAX and "BLOCK = 128" in TUNED_SOFTMAX
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    shutil.copy(ZOO / "softmax" / "kernel.cu", repo / "softmax.cu")
    spec = json.loads((ZOO / "softmax" / "launch.json").read_text())
    spec["source"] = "softmax.cu"
    root = str(repo)
    tools.scan_kernels(".", root=root)
    tools.write_reference("softmax", (ZOO / "softmax" / "reference.py").read_text(), spec=spec, root=root)
    assert tools.validate_reference("softmax", root=root)["status"] == "PASS"

    assert not tools.tune_port("softmax", "triton", root=root)["ok"]          # no port yet
    tools.write_port("softmax", "triton", TUNED_SOFTMAX, root=root)
    assert "verify_port first" in tools.tune_port("softmax", "triton", root=root)["summary"]
    assert tools.verify_port("softmax", "triton", root=root)["status"] == "PASS"

    res = tools.tune_port("softmax", "triton", root=root)
    assert res["ok"], res
    assert res["variants_total"] == 2 and res["variants_passed"] == 2
    assert all(r["status"] == "PASS" for r in res["rows"])
    ws = Workspace(repo)
    port = ws.port_path("softmax", "triton")
    if res["changed"]:
        assert f"BLOCK = {res['best']['BLOCK']}" in port.read_text()
        assert res["gain"] >= 1.03
    # Whatever happened, the stored port still PASSES as it is now.
    assert tools.list_status(root=root)["rows"][0]["status"] == "PASS"
    assert (port.parent / "tune" / "tune.json").is_file()

    hip = tools.tune_port("softmax", "hip", root=root)
    assert not hip["ok"]
