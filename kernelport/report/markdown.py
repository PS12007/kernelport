"""Markdown report: the same data as the HTML report, for PR comments and agent replies."""

from __future__ import annotations

from pathlib import Path

from kernelport.report.data import STATUS_ORDER, cell_status, collect
from kernelport.workspace import Workspace


def _cell(text) -> str:
    return str("" if text is None else text).replace("|", "\\|").replace("\n", " ")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _matrix_cell(k: dict, backend: str) -> str:
    status = cell_status(k, backend)[0]
    if status != "PASS":
        return status or "-"
    port = next(p for p in k["ports"] if p["backend"] == backend)
    notes = [", ".join(port["passed_on"])] if port["passed_on"] else []
    if port["speedup"] is not None:
        notes.append(f"{port['speedup']:.2f}x")
    return "PASS" + (f" ({_cell('; '.join(notes))})" if notes else "")


def render(d: dict, max_failure_lines: int = 12) -> str:
    """Markdown for report data from ``collect``."""
    c = d["port_counts"]
    lines = ["# kernelport report", "",
             f"**{d['headline']}**", "",
             f"{d['repo']}, generated {d['generated']}, hardware: {', '.join(d['hardware']) or 'none yet'}", "",
             "Port results: " + ", ".join(f"{c.get(s, 0)} {s}" for s in STATUS_ORDER if c.get(s) or s != "SKIPPED"),
             "",
             "PASS = current code verified on real hardware against a reference validated on the original CUDA kernel.",
             ""]
    if not d["kernels"]:
        return "\n".join(lines + ["No kernels registered yet."]) + "\n"
    started = [k for k in d["kernels"] if k["started"]]
    if started:
        lines.append("| kernel | reference | " + " | ".join(d["backends"]) + " |")
        lines.append("|---|---|" + "---|" * len(d["backends"]))
    for k in started:
        cells = [_matrix_cell(k, b) for b in d["backends"]]
        lines.append(f"| `{_cell(k['name'])}` | {k['reference']['state']} | " + " | ".join(cells) + " |")
    idle = [k for k in d["kernels"] if not k["started"]]
    if idle:
        names = ", ".join(f"`{_cell(k['name'])}`" for k in idle[:30]) + (", ..." if len(idle) > 30 else "")
        lines += ["", f"{_plural(len(idle), 'more kernel')} found, not started: {names}"]

    bench = [(k, p, b) for k in d["kernels"] for p in k["ports"] for b in p["bench"] if b.get("current")]
    if bench:
        lines += ["", "## Benchmarks", "",
                  "| kernel | backend | hardware | case | dtype | port ms | original ms | speedup |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, p, b in bench:
            orig = f"{b['original_ms']:.4f}" if b.get("original_ms") is not None else "-"
            speed = f"{b['speedup']:.2f}x" if b.get("speedup") is not None else "-"
            if not b.get("comparable", True):
                speed += " (not comparable)"
            lines.append(f"| `{_cell(k['name'])}` | {p['backend']} | {_cell(b.get('hardware'))} | {_cell(b.get('case'))} "
                         f"| {b.get('dtype')} | {b.get('port_ms', 0):.4f} | {orig} | {speed} |")

    attention = [(k, p) for k in d["kernels"] for p in k["ports"]
                 if any(r["status"] != "PASS" for r in p["rows"])]
    if attention:
        lines += ["", "## Needs attention", ""]
        for k, p in attention:
            for r in p["rows"]:
                if r["status"] == "PASS":
                    continue
                lines.append(f"- `{k['name']}` / {p['backend']} on {r['hardware']}: **{r['status']}**"
                             + (f" ({r['reason']})" if r["reason"] else "")
                             + f", {_plural(p['attempts']['total'], 'attempt')}")
            if p["failure_report"]:
                rep = p["failure_report"].splitlines()
                if len(rep) > max_failure_lines:
                    rep = rep[:max_failure_lines] + ["..."]
                lines += ["", "  ```", *("  " + ln for ln in rep), "  ```"]
    if d.get("pipelines"):
        lines += ["", "## Pipelines", "",
                  "Kernels run in a chain; every stage checked on what the stage before wrote. "
                  "Drift: the chain vs a float64 chain of references (information, not a verdict).", "",
                  "| pipeline | stages | backend | hardware | status | cases | drift |",
                  "|---|---|---|---|---|---|---|"]
        for p in d["pipelines"]:
            status = f"**{p['status']}**" + (f" ({_cell(p['reason'])})" if p["reason"] else "")
            drift = "<br>".join(_cell(x) for x in p.get("drift", []))
            lines.append(f"| `{_cell(p['pipeline'])}` | {p['stages']} | {p['backend']} | {_cell(p['hardware'])} "
                         f"| {status} | {p['cases']} | {drift} |")
    unfrozen = [k for k in started if k["reference"]["state"] != "frozen"]
    if unfrozen:
        lines += ["", "## References not validated yet", ""]
        lines += [f"- `{k['name']}`: {k['reference']['state']}" for k in unfrozen]
    return "\n".join(lines) + "\n"


def build_markdown(ws: Workspace) -> Path:
    """Write .kernelport/report.md and return its path."""
    path = ws.safe_path("report.md")
    path.write_bytes(render(collect(ws)).encode("utf-8"))
    return path
