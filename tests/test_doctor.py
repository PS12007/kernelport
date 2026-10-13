"""Doctor must never crash, and must report every backend with a reason."""

from kernelport.doctor import BACKENDS, run_doctor


def test_doctor_reports_every_backend():
    report = run_doctor()
    names = [b.name for b in report.backends]
    assert names == [b[0] for b in BACKENDS]
    for b in report.backends:
        assert b.reason, f"{b.name} has no reason"
        assert b.smoke in {"pass", "fail", "skipped"}
        if not b.available:
            assert b.smoke == "skipped"


def test_doctor_has_no_failing_smoke_tests():
    report = run_doctor()
    failed = [(b.name, b.detail) for b in report.backends if b.smoke == "fail"]
    assert not failed, failed


def test_doctor_report_is_json_serialisable():
    import json

    json.dumps(run_doctor().to_dict())
