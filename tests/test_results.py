import pytest

from kernelport.harness.results import CaseResult, RunResult


def _run(status="FAIL"):
    bad_output = {"name": "y", "passed": False, "shape": [5], "reason": "1/5 values outside tolerance",
                  "expected_slice": [1.0], "actual_slice": [2.0], "slice_start": 3}
    return RunResult(kernel="k", backend="triton", status=status, hardware="gpu",
                     cases=[CaseResult("tiny", "float32", True, "h"),
                            CaseResult("odd", "float16", False, "h2", outputs=[bad_output])])


def test_roundtrip(tmp_path):
    r = _run()
    r.save(tmp_path / "result.json")
    assert RunResult.load(tmp_path / "result.json") == r


def test_rejects_unknown_status():
    with pytest.raises(ValueError):
        _run(status="MOSTLY_PASS")


def test_failure_report_is_compact():
    rep = _run().failure_report()
    assert "FAIL (1/2 cases passed)" in rep
    assert "odd dtype float16" in rep
    assert "expected [1.0]" in rep and "actual   [2.0]" in rep
    assert len(rep) < 1000


def test_rejects_other_schema():
    d = _run().to_dict()
    d["schema_version"] = 999
    with pytest.raises(ValueError):
        RunResult.from_dict(d)
