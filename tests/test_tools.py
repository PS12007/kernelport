"""Every tool, called directly, including the rules the tools must enforce."""

import json
import shutil

import pytest

from conftest import BAD_PORTS, ZOO, needs_cuda, needs_triton
from kernelport import tools
from kernelport.workspace import MAX_ATTEMPTS, Workspace


@pytest.fixture
def repo(tmp_path):
    """A fresh user repo holding only original CUDA files (no references, no ports)."""
    r = tmp_path / "userrepo"
    (r / ".git").mkdir(parents=True)
    for k in ("relu", "vector_add"):
        (r / "kernels").mkdir(exist_ok=True)
        shutil.copy(ZOO / k / "kernel.cu", r / "kernels" / f"{k}.cu")
    return str(r)


def _spec(kernel):
    spec = json.loads((ZOO / kernel / "launch.json").read_text())
    spec["source"] = f"{kernel}.cu"
    return spec


def _ref(kernel):
    return (ZOO / kernel / "reference.py").read_text()


def _port(kernel):
    return (ZOO / kernel / "ports" / "triton.py").read_text()


def test_doctor_tool():
    res = tools.kernelport_doctor()
    assert "summary" in res and res["backends"]


def test_scan_and_read(repo):
    res = tools.scan_kernels(".", root=repo)
    assert res["ok"] and sorted(k["name"] for k in res["kernels"]) == ["relu", "vector_add"]
    ids = {k["name"]: k["kernel_id"] for k in res["kernels"]}
    assert ids["relu"] == "kernels/relu.cu::relu"
    rk = tools.read_kernel("relu", root=repo)
    assert rk["ok"] and "__global__" in rk["source"]
    assert rk["input_spec_origin"] == "suggested"
    assert [p["kind"] for p in rk["params"]] == ["input", "output", "scalar"]
    assert rk["reference_frozen"] is False


def test_scan_rejects_paths_outside_repo(repo, tmp_path):
    assert not tools.scan_kernels("..", root=repo)["ok"]
    assert not tools.scan_kernels(str(tmp_path), root=repo)["ok"]


def test_read_skill():
    res = tools.read_skill("workflow")
    assert res["ok"] and res["content"]
    assert not tools.read_skill("../../etc/passwd")["ok"]


def test_write_reference_validation_errors(repo):
    tools.scan_kernels(".", root=repo)
    assert not tools.write_reference("relu", "def nope(): pass\n", root=repo)["ok"]
    bad_spec = _spec("relu")
    bad_spec["params"][0]["kind"] = "bogus"
    assert not tools.write_reference("relu", _ref("relu"), spec=bad_spec, root=repo)["ok"]
    res = tools.write_reference("relu", _ref("relu"), root=repo)
    assert res["ok"] and res["has_spec"] is False


def test_verify_refused_before_freeze(repo):
    tools.scan_kernels(".", root=repo)
    tools.write_reference("relu", _ref("relu"), spec=_spec("relu"), root=repo)
    tools.write_port("relu", "triton", _port("relu"), root=repo)
    res = tools.verify_port("relu", "triton", root=repo)
    assert res["status"] == "REFUSED" and "not frozen" in res["error"]
    assert not tools.bench_port("relu", "triton", root=repo)["ok"]


def test_write_port_unknown_backend(repo):
    tools.scan_kernels(".", root=repo)
    assert not tools.write_port("relu", "cobol", "x", root=repo)["ok"]


def test_status_and_report_on_empty_repo(repo):
    assert tools.list_status(root=repo)["rows"] == []
    assert not tools.make_report(root=repo)["ok"]


@needs_cuda
def test_wrong_reference_does_not_freeze(repo):
    tools.scan_kernels(".", root=repo)
    tools.write_reference("relu", "import torch\ndef reference(x):\n    return torch.abs(x)\n",
                          spec=_spec("relu"), root=repo)
    res = tools.validate_reference("relu", root=repo)
    assert res["status"] == "FAIL" and "report" in res
    assert not Workspace(repo).is_frozen("relu")


def test_cli_verify_and_bench_take_a_kernel_argument(repo, monkeypatch):
    from typer.testing import CliRunner

    from kernelport.cli import app

    tools.scan_kernels(".", root=repo)
    monkeypatch.setenv("KERNELPORT_ROOT", repo)
    runner = CliRunner()
    for cmd in ("verify", "bench"):
        r = runner.invoke(app, [cmd, "relu"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, [cmd, "nosuch"])
        assert r.exit_code == 2 and "unknown kernel" in r.output, r.output
        assert runner.invoke(app, [cmd, "relu", "--kernel", "vector_add"]).exit_code == 2


@needs_cuda
def test_cli_validate_all_freezes_pending(repo, monkeypatch):
    from typer.testing import CliRunner

    from kernelport.cli import app

    tools.scan_kernels(".", root=repo)
    for k in ("relu", "vector_add"):
        tools.write_reference(k, _ref(k), spec=_spec(k), root=repo)
    monkeypatch.setenv("KERNELPORT_ROOT", repo)
    runner = CliRunner()
    r = runner.invoke(app, ["validate", "--all"])
    assert r.exit_code == 0, r.output
    ws = Workspace(repo)
    assert all(ws.is_frozen(k) for k in ("relu", "vector_add"))
    r = runner.invoke(app, ["validate", "--all"])
    assert r.exit_code == 0 and "nothing to validate" in r.output
    assert runner.invoke(app, ["validate"]).exit_code != 0


@needs_cuda
@needs_triton
def test_full_workflow(repo):
    """scan -> read -> write_reference -> validate -> write_port -> verify -> bench -> status -> report."""
    tools.scan_kernels(".", root=repo)
    assert tools.write_reference("relu", _ref("relu"), spec=_spec("relu"), root=repo)["ok"]
    val = tools.validate_reference("relu", root=repo)
    assert val["status"] == "PASS", val
    assert val["manifest"]["reference_hash"]

    assert tools.write_port("relu", "triton", _port("relu"), root=repo)["ok"]
    ver = tools.verify_port("relu", "triton", root=repo)
    assert ver["status"] == "PASS", ver
    assert ver["attempt"] == 1

    b = tools.bench_port("relu", "triton", {"dtype": "float16"}, root=repo)
    assert b["ok"] and b["port_ms"] > 0 and b["original_ms"] > 0
    assert not tools.bench_port("relu", "triton", {"tolerance": 1}, root=repo)["ok"]

    rows = {(r["kernel"], r["backend"]): r["status"] for r in tools.list_status(root=repo)["rows"]}
    assert rows == {("kernels/relu.cu::relu", "triton"): "PASS", ("kernels/vector_add.cu::vector_add", "-"): "SKIPPED"}

    rep = tools.make_report(root=repo)
    assert rep["ok"] and "PASS" in open(rep["path"], encoding="utf-8").read()

    # Editing the port after PASS: status drops to UNVERIFIED, bench refuses.
    tools.write_port("relu", "triton", _port("relu") + "\n# edit\n", root=repo)
    relu_rows = [r for r in tools.list_status(root=repo)["rows"] if r["kernel"].endswith("::relu")]
    assert [r["status"] for r in relu_rows] == ["UNVERIFIED"]
    assert not tools.bench_port("relu", "triton", root=repo)["ok"]


@needs_cuda
@needs_triton
def test_failures_are_logged_and_budget_enforced_in_message(repo):
    tools.scan_kernels(".", root=repo)
    tools.write_reference("vector_add", _ref("vector_add"), spec=_spec("vector_add"), root=repo)
    assert tools.validate_reference("vector_add", root=repo)["status"] == "PASS"
    tools.write_port("vector_add", "triton", (BAD_PORTS / "vector_add_sub.py").read_text(), root=repo)
    last = None
    for _ in range(MAX_ATTEMPTS):
        last = tools.verify_port("vector_add", "triton", root=repo)
        assert last["status"] == "FAIL" and "outside tolerance" in last["report"]
    assert last["attempts_remaining"] == 0
    assert "stop and report" in last["summary"]
    assert len(Workspace(repo).attempts("vector_add", "triton")) == MAX_ATTEMPTS


def test_no_tool_accepts_tolerances():
    import inspect

    for fn in tools.TOOLS:
        params = inspect.signature(fn).parameters
        assert not any("tol" in p for p in params), fn.__name__
