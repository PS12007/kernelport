"""Report: built from a synthetic workspace (no GPU), honest statuses, escaping, both formats agree."""

import json
import re
import shutil

import pytest

from conftest import ZOO
from kernelport import tools
from kernelport.analyzer.cuda_parser import scan
from kernelport.harness.results import CaseResult, RunResult
from kernelport.report import html as html_report
from kernelport.report import markdown as md_report
from kernelport.report.data import cell_status, collect
from kernelport.workspace import Workspace, sha256_file

HW = "Test GPU <sm_x>"


def _result(ws, kid, backend, status, port, cases_ok=(True, True), reason=""):
    _, ref, manifest = ws.frozen(kid)
    cases = []
    for i, ok in enumerate(cases_ok):
        c = CaseResult(f"case{i}", "float32", ok, "h")
        if not ok:
            c.outputs.append({"name": "y", "passed": False, "reason": "12 elements outside tolerance",
                              "shape": [7], "expected_slice": [1.0], "actual_slice": [2.0], "slice_start": 3})
        cases.append(c)
    return RunResult(kernel=kid, backend=backend, status=status, hardware=HW, reason=reason,
                     code_hash=sha256_file(port), reference_hash=manifest["reference_hash"],
                     tolerances={"float32": {"rtol": 1.3e-6, "atol": 1e-5, "reason": "default"}}, cases=cases)


def _freeze(ws, name):
    rec = ws.get(name)
    ws.write_spec(name, json.loads((ZOO / name / "launch.json").read_text()))
    ws.write_reference(name, (ZOO / name / "reference.py").read_text())
    ws.freeze(name, RunResult(kernel=name, backend="cuda", status="PASS", hardware=HW,
                              reference_hash=sha256_file(rec.dir / "reference.py"), code_hash="src",
                              cases=[CaseResult("c", "float32", True, "h")]))


@pytest.fixture
def ws(tmp_path):
    """relu: triton PASS + bench, cpu FAIL after 2 attempts. softmax: triton PASS then edited (stale).
    vector_add: not frozen, no ports."""
    repo = tmp_path / "repo"
    for k in ("relu", "softmax", "vector_add"):
        shutil.copytree(ZOO / k, repo / k, ignore=shutil.ignore_patterns("ports", "__pycache__"))
    w = Workspace(repo)
    for k in scan(repo):
        w.register(k)
    _freeze(w, "relu")
    _freeze(w, "softmax")

    port = w.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    r = _result(w, "relu", "triton", "PASS", port)
    w.log_attempt("relu", "triton", r)
    w.save_result("relu", "triton", r)
    (port.parent / "bench").mkdir()
    (port.parent / "bench" / "test-gpu.json").write_text(json.dumps(
        {"case": "big", "dtype": "float32", "port_ms": 0.5, "original_ms": 0.6, "speedup": 1.2,
         "hardware": HW, "comparable": True, "note": "", "method": "cuda_event", "code_hash": sha256_file(port)}))

    cport = w.write_port("relu", "cpu", "def run(x, y, n):\n    y[:] = x  # <script>alert(1)</script>\n")
    for _ in range(2):
        f = _result(w, "relu", "cpu", "FAIL", cport, cases_ok=(True, False))
        w.log_attempt("relu", "cpu", f)
    w.save_result("relu", "cpu", f)

    sport = w.write_port("softmax", "triton", "def run(x, y, rows, cols):\n    pass\n")
    s = _result(w, "softmax", "triton", "PASS", sport)
    w.log_attempt("softmax", "triton", s)
    w.save_result("softmax", "triton", s)
    w.write_port("softmax", "triton", "def run(x, y, rows, cols):\n    pass  # edited\n")
    return w


def _kernel(d, name):
    return next(k for k in d["kernels"] if k["name"] == name)


def test_collect_statuses_are_honest(ws):
    d = collect(ws)
    relu, softmax, vadd = (_kernel(d, n) for n in ("relu", "softmax", "vector_add"))
    assert cell_status(relu, "triton")[0] == "PASS"
    assert cell_status(relu, "cpu")[0] == "FAIL"
    assert cell_status(softmax, "triton")[0] == "UNVERIFIED"  # edited after its PASS
    assert cell_status(vadd, "triton") == ("", "no port")
    assert vadd["reference"]["state"] == "not frozen"
    assert relu["reference"]["state"] == "frozen" and relu["reference"]["validated_on"] == HW
    assert d["counts"] == {"PASS": 1, "FAIL": 1, "UNVERIFIED": 1, "SKIPPED": 1}
    assert d["backends"] == ["cpu", "triton"]


def test_attempts_failure_report_and_bench(ws):
    relu = _kernel(collect(ws), "relu")
    cpu = next(p for p in relu["ports"] if p["backend"] == "cpu")
    tri = next(p for p in relu["ports"] if p["backend"] == "triton")
    assert cpu["attempts"] == {"total": 2, "failed": 2, "first_pass_attempt": None, "history": ["FAIL", "FAIL"]}
    assert "outside tolerance" in cpu["failure_report"]
    assert tri["attempts"]["first_pass_attempt"] == 1 and tri["failure_report"] == ""
    assert tri["bench"][0]["current"] and tri["bench"][0]["speedup"] == 1.2


def test_bench_goes_stale_when_port_changes(ws):
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass  # v2\n")
    tri = next(p for p in _kernel(collect(ws), "relu")["ports"] if p["backend"] == "triton")
    assert not tri["bench"][0]["current"]
    assert "Benchmarks" not in md_report.render(collect(ws))  # stale numbers are not presented


def test_html_escapes_and_embeds_data(ws):
    page = html_report.render(collect(ws))
    assert "<script>alert" not in page and "&lt;sm_x&gt;" in page
    blob = re.search(r'<script type="application/json" id="kernelport-data">(.*?)</script>', page, re.S).group(1)
    assert "</" not in blob.replace("<\\/", "")
    data = json.loads(blob)
    assert data["counts"]["PASS"] == 1
    # Only stylesheet is inline; nothing loads from the network.
    assert not re.search(r'(src|href)="https?:', page)


def test_markdown_matches_html_matrix(ws):
    d = collect(ws)
    md = md_report.render(d)
    assert "| `relu` | frozen | FAIL | PASS (Test GPU <sm_x>; 1.20x) |" in md
    assert "| `softmax` | frozen | - | UNVERIFIED |" in md
    assert "1 more kernel found, not started: `vector_add`" in md and "| `vector_add`" not in md
    assert md.splitlines()[2] == ("**1 kernel ported and verified: 1 PASS result on 1 hardware target, "
                                  "against 2 references validated on the original CUDA.**")
    assert "## Needs attention" in md and "port changed since this result" in md
    assert "1.20x" in md


def test_html_shows_where_ports_passed_and_folds_idle_kernels(ws):
    d = collect(ws)
    assert [k["name"] for k in d["kernels"] if not k["started"]] == ["vector_add"]
    assert d["port_counts"] == {"PASS": 1, "FAIL": 1, "UNVERIFIED": 1, "SKIPPED": 0}
    page = html_report.render(d)
    matrix = page.split("<h2>Kernels")[1].split("</table>")[0]
    assert "vector_add" not in matrix and "1.20x vs CUDA" in matrix and "Test GPU &lt;sm_x&gt;" in matrix
    assert "1 more kernel found by the scan, not started" in page
    assert 'id="k-' in page and page.count("<details id=") == 2  # only started kernels get details


def test_short_hardware_names():
    from kernelport.report.data import short_hardware

    assert short_hardware("NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)") == "RTX 5060"
    assert short_hardware("NVIDIA GeForce RTX 5060 Laptop GPU (WebGPU/Vulkan)") == "RTX 5060 (Vulkan)"
    assert short_hardware("Apple M1 (Virtual) (Metal)") == "Apple M1 (Virtual)"
    assert short_hardware("llvmpipe (LLVM 20.1.2, 256 bits) (WebGPU/Vulkan)") == "llvmpipe (Vulkan)"
    assert short_hardware("AMD Instinct MI300X (gfx942)") == "Instinct MI300X"
    assert short_hardware("Intel(R) Core(TM) i7-14700HX (28 threads)") == "Intel Core i7-14700HX"


def test_make_report_tool_both_formats(ws):
    h = tools.make_report(root=ws.root)
    assert h["ok"] and h["path"].endswith("report.html") and h["counts"]["FAIL"] == 1
    m = tools.make_report("markdown", root=ws.root)
    assert m["ok"] and m["path"].endswith("report.md") and m["content"].startswith("# kernelport report")
    assert not tools.make_report("pdf", root=ws.root)["ok"]
