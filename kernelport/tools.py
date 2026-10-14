"""The kernelport tools. One implementation shared by the MCP server, the CLI and the
optional built-in agent, so all three behave identically.

Every tool returns a JSON-serialisable dict with at least ``summary`` (one or two
human-readable lines). Rules that protect correctness are enforced here, not in
the skill:

- verify_port refuses unless the reference is frozen (and untampered).
- Tolerances come from kernelport's backend configs; no tool accepts them.
- A frozen reference is only replaced by a passing validate_reference.
- Every verify_port call is logged to attempts.jsonl.
- All writes stay inside the repo's .kernelport/ workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernelport.backends.base import BACKEND_NAMES, get_backend
from kernelport.workspace import MAX_ATTEMPTS, Workspace, WorkspaceError, sha256_file, slugify

SKILL_FILES = {
    "workflow": "SKILL.md",
    "common_bugs": "common_bugs.md",
    **{b: f"backends/{b}.md" for b in BACKEND_NAMES},
}


def _ws(root: str | Path | None = None) -> Workspace:
    return Workspace(Path(root) if root else None)


def _err(msg: str, **extra: Any) -> dict:
    return {"ok": False, "error": msg, "summary": f"error: {msg}", **extra}


def skills_dir() -> Path:
    """Directory holding SKILL.md (installed package or repo checkout)."""
    here = Path(__file__).resolve().parent
    for d in (here / "_skills" / "kernelport", here.parent / "skills" / "kernelport"):
        if (d / "SKILL.md").is_file():
            return d
    raise FileNotFoundError("kernelport skills not found")


# --------------------------------------------------------------------- tools

def kernelport_doctor() -> dict:
    """Environment report: OS, GPU, installed extras, and each backend's availability with reason."""
    from kernelport.doctor import run_doctor

    rep = run_doctor()
    avail = [b.name for b in rep.backends if b.available]
    summary = (f"{rep.os}; GPU: {rep.gpu_name or 'none'} {rep.compute_capability}; "
               f"available: {', '.join(avail) or 'none'}")
    return {"ok": rep.ok, **rep.to_dict(), "summary": summary}


def scan_kernels(path: str = ".", root: str | None = None) -> dict:
    """Find CUDA kernels under ``path`` (inside the repo) and register them in the workspace."""
    from kernelport.analyzer.cuda_parser import scan

    ws = _ws(root)
    target = (ws.root / path).resolve()
    if target != ws.root and ws.root not in target.parents:
        return _err(f"path {path!r} is outside the repo {ws.root}")
    if not target.exists():
        return _err(f"path {path!r} does not exist")
    kernels = scan(target)
    rel_prefix = target.relative_to(ws.root).as_posix() if target != ws.root else ""
    rows = []
    for k in kernels:
        if target.is_file():
            k.file = target.relative_to(ws.root).as_posix()
        elif rel_prefix:
            k.file = f"{rel_prefix}/{k.file}"
        k.id = f"{k.file}::{k.id.split('::', 1)[1]}"
        ws.register(k)
        rows.append({"kernel_id": k.id, "name": k.name, "file": k.file, "line": k.line,
                     "params": [f"{p.name}:{p.kind}" for p in k.params], "features": k.features})
    return {"ok": True, "kernels": rows,
            "summary": f"found {len(rows)} kernels in {path}" + (": " + ", ".join(r["name"] for r in rows[:12]) if rows else "")}


def read_kernel(kernel_id: str, root: str | None = None) -> dict:
    """Source, parsed signature, launch sites, workspace state, and an input spec to start from."""
    from kernelport.analyzer.cuda_parser import scan
    from kernelport.analyzer.suggest import has_todo, suggest_spec

    ws = _ws(root)
    try:
        rec = ws.get(kernel_id)
    except WorkspaceError as exc:
        return _err(str(exc))
    src_file = ws.original_source(rec)
    found = [k for k in scan(src_file) if k.name == rec.name and k.line == rec.info["line"]] \
        if not rec.info["file"].endswith(".py") else []
    source = found[0].source if found else src_file.read_text(encoding="utf-8")
    info = found[0] if found else None
    spec_path = rec.dir / "launch.json"
    if spec_path.is_file():
        spec, spec_origin = json.loads(spec_path.read_text(encoding="utf-8")), "saved"
    elif info is not None:
        spec, spec_origin = suggest_spec(info), "suggested"
        spec["source"] = rec.info["file"].rsplit("/", 1)[-1]
    else:
        spec, spec_origin = None, "none"
    ports = sorted(p.name for p in (rec.dir / "ports").iterdir()) if (rec.dir / "ports").is_dir() else []
    out = {
        "ok": True, "kernel_id": rec.id, "name": rec.name, "file": rec.info["file"],
        "line": rec.info["line"], "source": source,
        # compact: the full analyzer record (read/written flags, base types) lives in kernel.json
        "params": [{"name": q["name"], "type": q["type"], "kind": q["kind"]} for q in rec.info["params"]],
        "template_params": rec.info["template_params"],
        "features": rec.info["features"], "launches": rec.info["launches"],
        "input_spec": spec, "input_spec_origin": spec_origin,
        "input_spec_has_todo": bool(spec) and has_todo(spec),
        "reference_frozen": ws.is_frozen(rec.id), "ports": ports,
    }
    out["summary"] = (f"{rec.id}: {len(rec.info['params'])} params, features [{', '.join(rec.info['features'])}]; "
                      f"input spec {spec_origin}" + (" (has TODOs)" if out["input_spec_has_todo"] else "")
                      + f"; reference {'frozen' if out['reference_frozen'] else 'not frozen'}")
    return out


def read_skill(name: str = "workflow") -> dict:
    """Return a skill file: 'workflow', 'common_bugs', or a backend name (triton, cpu, metal, webgpu, hip)."""
    if name not in SKILL_FILES:
        return _err(f"unknown skill {name!r}; choose from {', '.join(SKILL_FILES)}")
    path = skills_dir() / SKILL_FILES[name]
    if not path.is_file():
        return _err(f"skill {name!r} is not written yet", available=[n for n, f in SKILL_FILES.items()
                                                                   if (skills_dir() / f).is_file()])
    text = path.read_text(encoding="utf-8")
    return {"ok": True, "name": name, "content": text, "summary": f"skill {name}: {len(text.splitlines())} lines"}


def write_reference(kernel_id: str, code: str, spec: dict | None = None, root: str | None = None) -> dict:
    """Save a candidate PyTorch reference (and optionally the input spec). Does not freeze anything."""
    ws = _ws(root)
    try:
        if spec is not None:
            ws.write_spec(kernel_id, spec)
        path = ws.write_reference(kernel_id, code)
        rec = ws.get(kernel_id)
    except WorkspaceError as exc:
        return _err(str(exc))
    has_spec = (rec.dir / "launch.json").is_file()
    return {"ok": True, "path": str(path.relative_to(ws.root)), "has_spec": has_spec,
            "summary": "reference saved" + ("" if has_spec else "; no input spec yet: pass spec=...")
            + ". Next: validate_reference."}


def validate_reference(kernel_id: str, root: str | None = None) -> dict:
    """Run the ORIGINAL CUDA kernel against the candidate reference on every case. Freezes on PASS.

    Returns status PASS, FAIL (with a compact report) or UNAVAILABLE (no NVIDIA GPU + CuPy here).
    """
    from kernelport.harness import run_cuda
    from kernelport.harness.verify import run_cases
    from kernelport.hardware import gpu_name

    ws = _ws(root)
    try:
        rec = ws.get(kernel_id)
        spec, ref = ws.candidate(kernel_id)
    except WorkspaceError as exc:
        return _err(str(exc))
    ok, why = run_cuda.is_available()
    if not ok:
        return {"ok": False, "status": "UNAVAILABLE", "reason": why,
                "summary": f"cannot run the original CUDA kernel here: {why}"}
    src = ws.original_source(rec)
    result = run_cases(kernel=rec.name, backend="cuda", hardware=gpu_name(), spec=spec, reference_path=ref,
                       runner=lambda d: run_cuda.run_original(src, spec, d),
                       code_hash=run_cuda.source_hash(src))
    ws.save_validation(rec.id, result)
    out = {"ok": result.status == "PASS", "status": result.status,
           "cases_passed": result.num_passed, "cases_total": len(result.cases), "hardware": result.hardware}
    if result.status == "PASS":
        try:
            out["manifest"] = ws.freeze(rec.id, result)
        except WorkspaceError as exc:
            return _err(str(exc), status="FAIL")
        out["summary"] = (f"PASS: original CUDA matches the reference on {result.num_passed}/{len(result.cases)} "
                          f"cases on {result.hardware}. Reference frozen.")
    else:
        out["report"] = result.failure_report()
        out["summary"] = (f"FAIL: reference disagrees with the original CUDA kernel. Fix the REFERENCE "
                          f"(never the original) and validate again.")
    return out


def write_port(kernel_id: str, backend: str, code: str, root: str | None = None) -> dict:
    """Save a port for ``backend``. Earlier results for older code no longer count."""
    ws = _ws(root)
    try:
        path = ws.write_port(kernel_id, backend, code)
    except WorkspaceError as exc:
        return _err(str(exc))
    ok, why = get_backend(backend).is_available()
    note = "Next: verify_port." if ok else f"{backend} is not available here ({why}); the port is saved as UNVERIFIED."
    return {"ok": True, "path": str(path.relative_to(ws.root)), "backend_available": ok,
            "summary": f"port saved to {path.relative_to(ws.root).as_posix()}. {note}"}


def verify_port(kernel_id: str, backend: str, root: str | None = None) -> dict:
    """Run the port against the FROZEN reference on every input case. PASS, FAIL (+report) or UNVERIFIED."""
    from kernelport.zoo import verify_port as _verify

    ws = _ws(root)
    try:
        spec, ref, _ = ws.frozen(kernel_id)
        port = ws.port_path(kernel_id, backend)
    except WorkspaceError as exc:
        return _err(str(exc), status="REFUSED")
    if not port.is_file():
        return _err(f"no {backend} port saved for {kernel_id}; call write_port first", status="REFUSED")
    result = _verify(spec, ref, port, backend)
    entry = ws.log_attempt(kernel_id, backend, result)
    if result.status != "UNVERIFIED":
        ws.save_result(kernel_id, backend, result)
    failures = ws.failures_since_pass(kernel_id, backend)
    out = {"ok": result.status == "PASS", "status": result.status, "hardware": result.hardware,
           "cases_passed": result.num_passed, "cases_total": len(result.cases), "attempt": entry["attempt"],
           "failed_attempts": failures, "attempts_remaining": max(0, MAX_ATTEMPTS - failures)}
    if result.status == "PASS":
        out["summary"] = f"PASS: {result.num_passed}/{len(result.cases)} cases on {result.hardware}."
        if result.reason:
            out["summary"] += f" Note: {result.reason}."
    elif result.status == "FAIL":
        out["report"] = result.failure_report()
        out["summary"] = f"FAIL ({result.num_passed}/{len(result.cases)} cases passed), attempt {failures}/{MAX_ATTEMPTS}."
        if failures >= MAX_ATTEMPTS:
            out["summary"] += " Repair budget used up: stop and report this port as FAIL."
    else:
        out["reason"] = result.reason
        out["summary"] = f"{result.status}: {result.reason}"
    return out


def _passed_here(ws: Workspace, kernel_id: str, backend: str, port: Path) -> bool:
    """True if the port's current code has a stored PASS on this machine's hardware."""
    hw = get_backend(backend).hardware_name()
    return port.is_file() and any(r.hardware == hw and r.status == "PASS" and r.code_hash == sha256_file(port)
                                  for r in ws.results(kernel_id, backend))


def _default_dtype(spec, be) -> str:
    """float32 when the spec and backend allow it, else the spec's first dtype the backend runs."""
    runnable = [d for d in spec.dtypes if be.supports(d)]
    return "float32" if "float32" in runnable else (runnable or spec.dtypes)[0]


def bench_port(kernel_id: str, backend: str, config: dict | None = None, root: str | None = None) -> dict:
    """Time original vs port on this machine. Only for ports whose current code PASSED here.

    ``config`` may set {"dtype": "float16", "case": "<case name>"}; defaults: float32 (or the spec's first dtype this backend runs), the spec's bench case.
    """
    from kernelport.zoo import bench_port as _bench

    ws = _ws(root)
    config = config or {}
    unknown = set(config) - {"dtype", "case"}
    if unknown:
        return _err(f"unknown config keys {sorted(unknown)}; allowed: dtype, case")
    try:
        rec = ws.get(kernel_id)
        spec, _, _ = ws.frozen(kernel_id)
        port = ws.port_path(kernel_id, backend)
    except WorkspaceError as exc:
        return _err(str(exc))
    be = get_backend(backend)
    hw = be.hardware_name()
    if not _passed_here(ws, kernel_id, backend, port):
        return _err(f"no PASS for the current {backend} port on {hw}; run verify_port first")
    if "case" in config:
        try:
            spec.get_case(config["case"])
        except KeyError:
            return _err(f"unknown case {config['case']!r}")
        spec.bench_case = config["case"]
    dtype = config.get("dtype") or _default_dtype(spec, be)
    if dtype not in spec.dtypes or not be.supports(dtype):
        return _err(f"dtype {dtype} not in the spec or not supported by {backend}")
    timing = _bench(spec, ws.original_source(rec), port, backend, dtype)
    path = port.parent / "bench" / f"{slugify(hw, 64)}.json"
    ws.safe_path(*path.relative_to(ws.dir).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({**timing, "code_hash": sha256_file(port)}, indent=2).encode())
    lo, hi = timing["port_ms_range"]
    s = (f"{backend} {timing['port_ms']:.4f} ms (runs {lo:.4f}-{hi:.4f}) on {hw} "
         f"({timing['case']}, {dtype}, median of {timing['rounds']} runs)")
    if timing["original_ms"] is not None:
        olo, ohi = timing["original_ms_range"]
        s += (f"; original CUDA {timing['original_ms']:.4f} ms (runs {olo:.4f}-{ohi:.4f}); "
              f"speedup {timing['speedup']:.2f}x")
    if timing["noisy"]:
        s += ". Runs differed by more than 20%: the GPU's clocks or load changed; treat the speedup as approximate"
    if not timing["comparable"]:
        s += f". Not comparable to native GPU numbers: {timing['note']}"
    return {"ok": True, **timing, "summary": s}


def tune_port(kernel_id: str, backend: str, dtype: str | None = None, root: str | None = None) -> dict:
    """Try every combination in the port's TUNE dict; keep the fastest variant that PASSES every case.

    Only for Python ports whose current code PASSED here. Each variant is verified against the
    frozen reference before it is timed; passing variants are then timed in interleaved rounds.
    A variant replaces the port only if its median is at least 3% faster and every one of its
    rounds beat every round of the current configuration; the new port is verified again so the
    stored PASS matches the new code.
    """
    import statistics

    from kernelport.tune import TUNE_ROUNDS, TUNE_WARMUP, TuneError, render, variants
    from kernelport.zoo import time_port as _time, verify_port as _verify

    ws = _ws(root)
    try:
        spec, ref, _ = ws.frozen(kernel_id)
        port = ws.port_path(kernel_id, backend)
    except WorkspaceError as exc:
        return _err(str(exc))
    be = get_backend(backend)
    hw = be.hardware_name()
    if be.config.extension != ".py":
        return _err(f"tuning works on Python ports; {backend} ports are {be.config.extension} files")
    if not _passed_here(ws, kernel_id, backend, port):
        return _err(f"no PASS for the current {backend} port on {hw}; run verify_port first")
    dtype = dtype or _default_dtype(spec, be)
    if dtype not in spec.dtypes or not be.supports(dtype):
        return _err(f"dtype {dtype} not in the spec or not supported by {backend}")
    code = port.read_bytes().decode("utf-8")
    try:
        combos, current = variants(code)
    except TuneError as exc:
        return _err(str(exc))

    tune_dir = port.parent / "tune"
    rows = []
    for i, params in enumerate(combos):
        vcode = render(code, params)
        vpath = ws.safe_path(*(tune_dir / f"v{i:02d}.py").relative_to(ws.dir).parts)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        vpath.write_bytes(vcode.encode("utf-8"))
        res = _verify(spec, ref, vpath, backend)
        row = {"params": params, "status": res.status, "current": params == current,
               "cases_passed": res.num_passed, "cases_total": len(res.cases), "port_ms": None,
               "code": vcode, "path": vpath, "rounds": []}
        if res.status == "FAIL":
            row["failure"] = res.failure_report().splitlines()[0]
        rows.append(row)
    passing = [r for r in rows if r["status"] == "PASS"]
    # Round-robin timing on one shared input set: clock ramp-up and thermal drift hit every
    # variant alike, and the GPU never idles while inputs are regenerated.
    if len(passing) > 1:
        from kernelport.harness.inputs import make_case

        case = spec.get_case(spec.bench_case) if spec.bench_case else spec.cases[-1]
        data = make_case(spec, case, dtype)
        for _ in range(TUNE_ROUNDS):
            for r in passing:
                r["rounds"].append(_time(spec, r["path"], backend, dtype, data=data,
                                         warmup=TUNE_WARMUP)[0].median_ms)
    for r in passing:
        r["port_ms"] = statistics.median(r["rounds"]) if r["rounds"] else None
    base = next((r for r in rows if r["current"]), None)
    for r in rows:  # variants are reproducible from TUNE; only tune.json is kept
        r["path"].unlink(missing_ok=True)
    out_rows = [{k: v for k, v in r.items() if k not in ("code", "path")} for r in rows]
    ws.safe_path(*(tune_dir / "tune.json").relative_to(ws.dir).parts).write_bytes(
        json.dumps({"hardware": hw, "dtype": dtype, "rows": out_rows}, indent=2).encode())
    if base is None or base["status"] != "PASS":
        return {"ok": False, "rows": out_rows,
                "summary": "the current configuration did not pass as a variant; port left unchanged"}
    if len(passing) == 1:
        return {"ok": True, "rows": out_rows, "changed": False, "variants_passed": 1, "variants_total": len(rows),
                "summary": f"kept the current port: it is the only variant that passed (of {len(rows)})"}
    best = min(passing, key=lambda r: r["port_ms"])
    gain = base["port_ms"] / best["port_ms"]
    out = {"ok": True, "rows": out_rows, "hardware": hw, "dtype": dtype, "best": best["params"],
           "best_ms": best["port_ms"], "current_ms": base["port_ms"], "gain": gain,
           "variants_passed": len(passing), "variants_total": len(rows)}
    # Change the port only for a clear win: >= 3% on the median AND every round of the winner
    # faster than every round of the current configuration.
    if best is base or gain < 1.03 or max(best["rounds"]) >= min(base["rounds"]):
        out["changed"] = False
        if best is base:
            why = "the current configuration was the fastest"
        elif gain < 1.03:
            why = f"the best variant {best['params']} was only {100 * (gain - 1):.0f}% faster (3% needed)"
        else:
            why = (f"the best variant {best['params']} had a {100 * (gain - 1):.0f}% faster median, but its "
                   f"{TUNE_ROUNDS} timing rounds ({min(best['rounds']):.4f}-{max(best['rounds']):.4f} ms) "
                   f"overlapped the current port's ({min(base['rounds']):.4f}-{max(base['rounds']):.4f} ms), "
                   "so the difference is within timing noise here")
        out["summary"] = (f"kept the current port ({base['port_ms']:.4f} ms): {why}; "
                          f"{len(passing)}/{len(rows)} variants passed")
        return out
    ws.write_port(kernel_id, backend, best["code"])
    ver = verify_port(kernel_id, backend, root=str(ws.root))
    if ver["status"] != "PASS":  # pragma: no cover - same code just passed as a variant
        ws.write_port(kernel_id, backend, code)
        verify_port(kernel_id, backend, root=str(ws.root))
        return _err(f"tuned variant did not pass on re-verification ({ver['summary']}); port restored")
    out["changed"] = True
    out["summary"] = (f"tuned: {best['params']} is {gain:.2f}x faster than {current} on {hw} "
                      f"({best['port_ms']:.4f} vs {base['port_ms']:.4f} ms, {dtype}); port updated and "
                      f"verified again ({len(passing)}/{len(rows)} variants passed)")
    return out


def list_status(root: str | None = None) -> dict:
    """Table of kernel x backend x hardware statuses (PASS, FAIL, UNVERIFIED, SKIPPED)."""
    ws = _ws(root)
    rows = ws.status() if ws.dir.is_dir() else []
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "workspace is empty"
    return {"ok": True, "rows": rows, "counts": counts, "summary": summary}


def make_report(format: str = "html", root: str | None = None) -> dict:
    """Write the report to .kernelport/report.html (format="html") or report.md (format="markdown").

    The Markdown variant is also returned inline as ``content``, ready to paste into a reply or PR.
    """
    from kernelport.report.html import build_report
    from kernelport.report.markdown import build_markdown

    if format not in ("html", "markdown"):
        return _err(f"unknown format {format!r}; choose html or markdown")
    ws = _ws(root)
    if not ws.dir.is_dir():
        return _err("workspace is empty; nothing to report")
    rows = ws.status()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    tally = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    if format == "markdown":
        path = build_markdown(ws)
        return {"ok": True, "path": str(path), "counts": counts,
                "content": path.read_text(encoding="utf-8"), "summary": f"report written to {path} ({tally})"}
    path = build_report(ws)
    return {"ok": True, "path": str(path), "counts": counts, "summary": f"report written to {path} ({tally})"}


TOOLS = [kernelport_doctor, scan_kernels, read_kernel, read_skill, write_reference, validate_reference,
         write_port, verify_port, bench_port, tune_port, list_status, make_report]

