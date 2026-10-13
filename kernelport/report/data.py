"""Collect everything a report shows from the workspace, once, as plain data.

Both renderers (HTML and Markdown) read this structure, so they always agree.
Statuses come from ``Workspace.status()``, which already downgrades stale
results (port or reference changed) to UNVERIFIED; benchmarks are only shown
as current when their code hash matches the port on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernelport import __version__
from kernelport.backends.base import get_backend
from kernelport.harness.results import RunResult
from kernelport.workspace import Workspace, WorkspaceError, sha256_file

STATUS_ORDER = ("PASS", "FAIL", "UNVERIFIED", "SKIPPED")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _reference_info(ws: Workspace, kid: str) -> dict:
    """Frozen-reference provenance, or why there is none."""
    try:
        _, _, manifest = ws.frozen(kid)
    except WorkspaceError as exc:
        state = "not frozen" if "not frozen" in str(exc) else "tampered"
        return {"state": state, "detail": str(exc)}
    return {"state": "frozen", "validated_on": manifest.get("validated_on", ""),
            "validated_at": manifest.get("validated_at", ""),
            "cases_passed": manifest.get("cases_passed"),
            "reference_hash": manifest.get("reference_hash", "")[:12]}


def _attempt_stats(attempts: list[dict]) -> dict:
    first_pass = next((a["attempt"] for a in attempts if a["status"] == "PASS"), None)
    return {"total": len(attempts),
            "failed": sum(a["status"] == "FAIL" for a in attempts),
            "first_pass_attempt": first_pass,
            "history": [a["status"] for a in attempts]}


def _bench_rows(port: Path) -> list[dict]:
    """Stored benchmarks for a port; ``current`` is False when the port changed since."""
    bdir = port.parent / "bench"
    if not bdir.is_dir():
        return []
    code_hash = sha256_file(port) if port.is_file() else None
    rows = []
    for p in sorted(bdir.glob("*.json")):
        b = _load_json(p)
        if not b:
            continue
        b["current"] = code_hash is not None and b.get("code_hash") == code_hash
        rows.append(b)
    return rows


def _latest_failure(results: list[RunResult], rows: list[dict]) -> str:
    """Failure report of a FAIL that still applies (status row says FAIL)."""
    failing_hw = {r["hardware"] for r in rows if r["status"] == "FAIL"}
    for r in results:
        if r.status == "FAIL" and r.hardware in failing_hw:
            return r.failure_report()
    return ""


def collect(ws: Workspace) -> dict[str, Any]:
    """Report data for the whole workspace."""
    status_rows = ws.status()
    by_kernel: dict[str, list[dict]] = {}
    for r in status_rows:
        by_kernel.setdefault(r["kernel"], []).append(r)

    kernels = []
    backends_seen: list[str] = []
    for kid in ws.kernel_ids():
        rec = ws.get(kid)
        rows = by_kernel.get(kid, [])
        ports = []
        for b in sorted({r["backend"] for r in rows if r["backend"] != "-"}):
            if b not in backends_seen:
                backends_seen.append(b)
            port = ws.port_path(kid, b)
            brows = [r for r in rows if r["backend"] == b]
            results = ws.results(kid, b)
            tolerances = {}
            for res in results:
                tolerances.update(res.tolerances)
            cfg = get_backend(b).config
            ports.append({
                "backend": b,
                "rows": brows,
                "attempts": _attempt_stats(ws.attempts(kid, b)),
                "bench": _bench_rows(port),
                "failure_report": _latest_failure(results, brows),
                "tolerances": tolerances,
                "timing_comparable": cfg.timing_comparable,
                "timing_note": cfg.timing_note,
                "port_lines": len(port.read_text(encoding="utf-8").splitlines()) if port.is_file() else 0,
            })
        kernels.append({
            "id": kid, "name": rec.name, "file": rec.info.get("file", ""), "line": rec.info.get("line"),
            "features": rec.info.get("features", []),
            "reference": _reference_info(ws, kid),
            "ports": ports,
        })

    counts = {s: 0 for s in STATUS_ORDER}
    for r in status_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    hardware = sorted({r["hardware"] for r in status_rows if r["hardware"] not in ("-", "")})
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "repo": ws.root.name,
        "hardware": hardware,
        "backends": sorted(backends_seen),
        "counts": counts,
        "kernels": kernels,
        "rows": status_rows,
    }


def cell_status(kernel: dict, backend: str) -> tuple[str, str]:
    """Overall (status, tooltip) for one kernel x backend matrix cell.

    Best status across machines wins (a PASS on one machine is a PASS for that
    machine); the tooltip lists every machine's status.
    """
    port = next((p for p in kernel["ports"] if p["backend"] == backend), None)
    if port is None:
        return "", "no port"
    rows = port["rows"]
    best = min((r["status"] for r in rows), key=STATUS_ORDER.index, default="SKIPPED")
    tip = "; ".join(f"{r['hardware']}: {r['status']}" + (f" ({r['reason']})" if r["reason"] else "")
                    for r in rows)
    return best, tip
