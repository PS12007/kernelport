"""Markdown report: the same data as the HTML report, for PR comments and agent replies."""

from __future__ import annotations

from pathlib import Path

from kernelport.report.data import STATUS_ORDER, cell_status, collect
from kernelport.workspace import Workspace


def _cell(text) -> str:
    return str("" if text is None else text).replace("|", "\\|").replace("\n", " ")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def render(d: dict, max_failure_lines: int = 12) -> str:
    """Markdown for report data from ``collect``."""
    c = d["counts"]
    lines = ["# kernelport report", "",
             f"{d['repo']}, generated {d['generated']}, hardware: {', '.join(d['hardware']) or 'none yet'}", "",
             "**" + ", ".join(f"{c.get(s, 0)} {s}" for s in STATUS_ORDER) + "**", "",
             "PASS = current code verified on real hardware against a reference validated on the original CUDA kernel.",
             ""]
    if not d["kernels"]:
        return "\n".join(lines + ["No kernels registered yet."]) + "\n"
    lines.append("| kernel | reference | " + " | ".join(d["backends"]) + " |")
    lines.append("|---|---|" + "---|" * len(d["backends"]))
    for k in d["kernels"]:
        cells = [cell_status(k, b)[0] or "-" for b in d["backends"]]
        lines.append(f"| `{_cell(k['name'])}` | {k['reference']['state']} | " + " | ".join(cells) + " |")

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
    unfrozen = [k for k in d["kernels"] if k["reference"]["state"] != "frozen"]
    if unfrozen:
        lines += ["", "## References not frozen", ""]
        lines += [f"- `{k['name']}`: {k['reference']['state']}" for k in unfrozen]
    return "\n".join(lines) + "\n"


def build_markdown(ws: Workspace) -> Path:
    """Write .kernelport/report.md and return its path."""
    path = ws.safe_path("report.md")
    path.write_bytes(render(collect(ws)).encode("utf-8"))
    return path
