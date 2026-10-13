"""Minimal HTML status report (expanded in M10)."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from kernelport.workspace import Workspace

STATUS_COLORS = {"PASS": "#1a7f37", "FAIL": "#cf222e", "UNVERIFIED": "#9a6700", "SKIPPED": "#6e7781"}


def build_report(ws: Workspace) -> Path:
    """Write .kernelport/report.html and return its path."""
    rows = ws.status()
    body = []
    for r in rows:
        color = STATUS_COLORS.get(r["status"], "#000")
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in ("kernel", "backend", "hardware"))
            + f'<td style="color:{color};font-weight:600">{r["status"]}</td>'
            + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in ("cases", "reason")) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>kernelport report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #d0d7de;padding:.3rem .6rem;text-align:left}}</style></head><body>
<h1>kernelport report</h1><p>Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}.
Only PASS means verified on real hardware against a frozen reference.</p>
<table><tr><th>kernel</th><th>backend</th><th>hardware</th><th>status</th><th>cases</th><th>reason</th></tr>
{''.join(body)}</table>
<script type="application/json" id="data">{html.escape(json.dumps(rows))}</script></body></html>"""
    path = ws.safe_path("report.html")
    path.write_bytes(page.encode("utf-8"))
    return path
