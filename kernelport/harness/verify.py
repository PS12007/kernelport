"""Run something (the original kernel or a port) on every case and compare to the reference.

This is the single code path behind both reference validation and port
verification, so both are held to the same checks.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import torch

from kernelport.harness.compare import compare_tensor
from kernelport.harness.inputs import TORCH_DTYPES, CaseData, make_case
from kernelport.harness.reference import file_hash, run_reference
from kernelport.harness.results import CaseResult, RunResult
from kernelport.harness.spec import LaunchSpec
from kernelport.harness.tolerances import Tolerance, tolerance_for

Runner = Callable[[CaseData], dict[str, torch.Tensor]]
MAX_ERROR_CHARS = 1200


def _user_frames(exc: BaseException) -> str:
    """Traceback lines from the port/reference file only (not kernelport, not libraries)."""
    pkg_dir = Path(__file__).resolve().parent.parent
    keep = []
    for fr in traceback.extract_tb(exc.__traceback__):
        f = Path(fr.filename)
        if "site-packages" in f.parts or fr.filename.startswith("<") or pkg_dir in f.resolve().parents:
            continue
        keep.append(f"  {f.name}, line {fr.lineno}, in {fr.name}: {(fr.line or '').strip()}")
    return "\n".join(keep[-3:])


def _short_error(exc: BaseException) -> str:
    """Exception text plus user-code frames, capped so it fits an agent's context."""
    msg = str(exc).split("\nTraceback (most recent call last)")[0].strip()
    text = f"{type(exc).__name__}: {msg}"
    frames = _user_frames(exc)
    full = f"{text}\n{frames}" if frames else text
    return full if len(full) <= MAX_ERROR_CHARS else full[:MAX_ERROR_CHARS] + " ...[truncated]"


def run_cases(*, kernel: str, backend: str, hardware: str, spec: LaunchSpec,
              reference_path: Path, runner: Runner, code_hash: str,
              tol_overrides: dict[str, Tolerance] | None = None,
              stop_after_failures: int | None = None,
              supported_dtypes: tuple[str, ...] | None = None) -> RunResult:
    """Run ``runner`` on every (case, dtype) and compare against the reference.

    Cases whose dtype is not in ``supported_dtypes`` are skipped and named in
    the result's reason. Status is PASS only if every remaining case ran and
    every output matched; if nothing remains, status is SKIPPED.
    """
    cases: list[CaseResult] = []
    used_tols: dict[str, dict] = {}
    failures = 0
    todo = [(c, t) for c, t in spec.iter_cases() if supported_dtypes is None or t in supported_dtypes]
    skipped = sorted({t for _, t in spec.iter_cases()} - {t for _, t in todo})
    for case, dtype in todo:
        data = make_case(spec, case, dtype)
        cr = CaseResult(case.name, dtype, False, data.input_hash)
        try:
            expected = run_reference(reference_path, spec, data)
        except Exception as exc:
            cr.error = "reference failed: " + _short_error(exc)
            cases.append(cr)
            failures += 1
            if stop_after_failures and failures >= stop_after_failures:
                break
            continue
        try:
            actual = runner(data)
        except Exception as exc:
            err = _short_error(exc)
            prev = next((c for c in cases if c.error == err or c.error.startswith(err)), None)
            cr.error = f"same error as {prev.case}/{prev.dtype}" if prev else err
            actual = None
        if actual is not None:
            ok = True
            for p in spec.params:
                if not p.is_output:
                    continue
                out_dtype = p.resolved_dtype(dtype)
                tol = tolerance_for(out_dtype, tol_overrides)
                used_tols[out_dtype] = {"rtol": tol.rtol, "atol": tol.atol, "reason": tol.reason}
                if p.name not in actual:
                    cr.outputs.append({"name": p.name, "passed": False, "reason": "output missing",
                                       "shape": [], "expected_slice": [], "actual_slice": [],
                                       "slice_start": 0})
                    ok = False
                    continue
                tc = compare_tensor(p.name, actual[p.name], expected[p.name],
                                    TORCH_DTYPES[out_dtype], tol)
                cr.outputs.append(tc.to_dict())
                ok = ok and tc.passed
            cr.passed = ok
        cases.append(cr)
        if not cr.passed:
            failures += 1
            if stop_after_failures and failures >= stop_after_failures:
                break

    all_ran = len(cases) == len(todo)
    if not todo:
        status = "SKIPPED"
    else:
        status = "PASS" if all_ran and all(c.passed for c in cases) else "FAIL"
    notes = []
    if not all_ran:
        notes.append(f"stopped early after {failures} failing cases")
    if skipped:
        notes.append(f"dtypes not supported by this backend, not tested: {', '.join(skipped)}")
    reason = "; ".join(notes)
    return RunResult(kernel=kernel, backend=backend, status=status, hardware=hardware,
                     reason=reason, code_hash=code_hash, reference_hash=file_hash(reference_path),
                     spec_hash=spec.hash(), tolerances=used_tols, cases=cases)
