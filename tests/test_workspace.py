"""Workspace: path safety, freezing rules, tamper detection, attempt logs, honest status."""

import json
import os
import shutil

import pytest

from conftest import ZOO
from kernelport.analyzer.cuda_parser import scan
from kernelport.harness.results import CaseResult, RunResult
from kernelport.workspace import Workspace, WorkspaceError, sha256_file


@pytest.fixture
def ws(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ZOO / "relu", repo / "relu")
    w = Workspace(repo)
    for k in scan(repo):
        w.register(k)
    return w


def _spec():
    return json.loads((ZOO / "relu" / "launch.json").read_text())


REF = (ZOO / "relu" / "reference.py").read_text()


def _validation(ws, status="PASS", backend="cuda"):
    rec = ws.get("relu")
    return RunResult(kernel="relu", backend=backend, status=status, hardware="Test GPU",
                     reference_hash=sha256_file(rec.dir / "reference.py"), code_hash="src",
                     cases=[CaseResult("tiny", "float32", status == "PASS", "h")])


def _ready(ws):
    ws.write_spec("relu", _spec())
    ws.write_reference("relu", REF)


def test_register_and_lookup_by_name(ws):
    assert ws.kernel_ids() == ["relu/kernel.cu::relu"]
    assert ws.get("relu").id == "relu/kernel.cu::relu"
    assert ws.get("relu/kernel.cu::relu").slug.startswith("relu-")
    with pytest.raises(WorkspaceError, match="unknown kernel"):
        ws.get("nope")


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "a\\b", "C:", ".", ""])
def test_safe_path_rejects_escapes(ws, bad):
    with pytest.raises(WorkspaceError):
        ws.safe_path("kernels", bad)


def test_unknown_backend_rejected(ws):
    with pytest.raises(WorkspaceError, match="unknown backend"):
        ws.write_port("relu", "../../evil", "x = 1\n")


def test_symlink_escape_rejected(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = ws.dir / "kernels" / ws.get("relu").slug / "ports"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    with pytest.raises(WorkspaceError, match="escapes"):
        ws.write_port("relu", "triton", "x = 1\n")
    assert not any(outside.iterdir())


def test_all_writes_stay_inside_workspace(ws):
    _ready(ws)
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    before = {p for p in ws.root.rglob("*") if WORKSPACE_PART not in p.parts}
    ws.freeze("relu", _validation(ws))
    after = {p for p in ws.root.rglob("*") if WORKSPACE_PART not in p.parts}
    assert before == after


WORKSPACE_PART = ".kernelport"


def test_spec_with_todo_rejected(ws):
    spec = _spec()
    spec["cases"][0]["dims"]["n"] = "TODO"
    with pytest.raises(WorkspaceError, match="TODO"):
        ws.write_spec("relu", spec)


def test_invalid_spec_rejected(ws):
    spec = _spec()
    spec["params"][0]["kind"] = "sideways"
    with pytest.raises(WorkspaceError, match="invalid spec"):
        ws.write_spec("relu", spec)


@pytest.mark.parametrize("code,msg", [
    ("def reference(x:\n", "syntax"),
    ("def other(x):\n    return x\n", "def reference"),
    ("", "empty"),
])
def test_bad_reference_rejected(ws, code, msg):
    with pytest.raises(WorkspaceError, match=msg):
        ws.write_reference("relu", code)


def test_not_frozen_until_validated(ws):
    _ready(ws)
    assert not ws.is_frozen("relu")
    with pytest.raises(WorkspaceError, match="not frozen"):
        ws.frozen("relu")


@pytest.mark.parametrize("status,backend", [("FAIL", "cuda"), ("UNVERIFIED", "cuda"), ("PASS", "triton")])
def test_only_passing_cuda_validation_freezes(ws, status, backend):
    _ready(ws)
    with pytest.raises(WorkspaceError):
        ws.freeze("relu", _validation(ws, status, backend))
    assert not ws.is_frozen("relu")


def test_freeze_and_candidate_edits_do_not_change_frozen(ws):
    _ready(ws)
    manifest = ws.freeze("relu", _validation(ws))
    assert manifest["validated_on"] == "Test GPU"
    ws.write_reference("relu", REF + "\n# edited candidate\n")
    _, ref_path, m2 = ws.frozen("relu")
    assert ref_path.read_text() == REF
    assert m2 == manifest


def test_freeze_refuses_if_reference_changed_during_validation(ws):
    _ready(ws)
    v = _validation(ws)
    ws.write_reference("relu", REF + "\n# changed\n")
    with pytest.raises(WorkspaceError, match="changed during validation"):
        ws.freeze("relu", v)


def test_tampered_frozen_reference_detected(ws):
    _ready(ws)
    ws.freeze("relu", _validation(ws))
    _, ref_path, _ = ws.frozen("relu")
    ref_path.write_text("def reference(x):\n    return x * 0\n")
    with pytest.raises(WorkspaceError, match="modified after validation"):
        ws.frozen("relu")
    assert not ws.is_frozen("relu")


def test_attempt_log_and_repair_budget(ws):
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    for status in ["FAIL", "FAIL", "PASS", "FAIL"]:
        ws.log_attempt("relu", "triton", RunResult(kernel="relu", backend="triton", status=status,
                                                     hardware="gpu", code_hash="c"))
    att = ws.attempts("relu", "triton")
    assert [a["attempt"] for a in att] == [1, 2, 3, 4]
    assert all({"timestamp", "code_hash", "status"} <= set(a) for a in att)
    assert ws.failures_since_pass("relu", "triton") == 1


def _port_result(ws, status, hw, code_hash=None, ref_hash=None):
    _, _, manifest = ws.frozen("relu")
    return RunResult(kernel="relu", backend="triton", status=status, hardware=hw,
                     code_hash=code_hash or sha256_file(ws.port_path("relu", "triton")),
                     reference_hash=ref_hash or manifest["reference_hash"],
                     cases=[CaseResult("tiny", "float32", status == "PASS", "h")])


def test_status_per_hardware_and_staleness(ws):
    _ready(ws)
    ws.freeze("relu", _validation(ws))
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    rows = ws.status()
    assert [(r["backend"], r["status"]) for r in rows] == [("triton", "UNVERIFIED")]

    ws.save_result("relu", "triton", _port_result(ws, "PASS", "NVIDIA RTX 5060 (sm_120)"))
    ws.save_result("relu", "triton", _port_result(ws, "FAIL", "AMD MI300X (gfx942)"))
    rows = {r["hardware"]: r["status"] for r in ws.status()}
    assert rows == {"NVIDIA RTX 5060 (sm_120)": "PASS", "AMD MI300X (gfx942)": "FAIL"}

    # Editing the port invalidates every earlier result.
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass  # v2\n")
    rows = ws.status()
    assert {r["status"] for r in rows} == {"UNVERIFIED"}
    assert all("port changed" in r["reason"] for r in rows)


def test_status_pass_with_other_reference_is_unverified(ws):
    _ready(ws)
    ws.freeze("relu", _validation(ws))
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    ws.save_result("relu", "triton", _port_result(ws, "PASS", "gpu", ref_hash="0" * 64))
    (row,) = ws.status()
    assert row["status"] == "UNVERIFIED"


def test_status_never_upgrades_to_pass(ws):
    """No combination of stored results may display PASS unless the stored result is PASS."""
    _ready(ws)
    ws.freeze("relu", _validation(ws))
    ws.write_port("relu", "triton", "def run(x, y, n):\n    pass\n")
    for st in ("FAIL", "UNVERIFIED", "SKIPPED"):
        ws.save_result("relu", "triton", _port_result(ws, st, f"hw-{st}"))
    assert "PASS" not in {r["status"] for r in ws.status()}


def test_python_embedded_kernel_snapshot(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    shutil.copy(ZOO.parent / "tests" / "fixtures" / "embedded_kernel.py", repo / "k.py")
    w = Workspace(repo)
    rec = w.register(scan(repo)[0])
    src = w.original_source(rec)
    assert src.name == "original.cu" and "__global__" in src.read_text()


def test_files_written_byte_exact(ws):
    _ready(ws)
    raw = (ws.get("relu").dir / "reference.py").read_bytes()
    assert raw == REF.encode()  # no CRLF translation on Windows
