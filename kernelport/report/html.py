"""Self-contained HTML report: summary, kernel x backend matrix, per-kernel details.

No external scripts, fonts or styles, so the file opens offline and can be
attached to a PR or emailed. Every value from the workspace is escaped.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from kernelport.report.data import STATUS_ORDER, cell_status, collect
from kernelport.workspace import Workspace

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa;
--pass:#1a7f37;--pass-bg:#dafbe1;--fail:#cf222e;--fail-bg:#ffebe9;
--unv:#9a6700;--unv-bg:#fff8c5;--skip:#59636e;--skip-bg:#eff2f5;--code:#eff2f5;--link:#0969da}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d1117;--fg:#e6edf3;
--muted:#9198a1;--line:#3d444d;--card:#151b23;--pass:#3fb950;--pass-bg:#12261e;--fail:#f85149;
--fail-bg:#2d1214;--unv:#d29922;--unv-bg:#272115;--skip:#9198a1;--skip-bg:#212830;--code:#151b23;
--link:#4493f8}}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
.headline{font-size:1.05rem;margin:12px 0 4px}
.cell .small{display:block;margin-top:2px}
.speed{font-variant-numeric:tabular-nums;white-space:nowrap}
.idle summary{color:var(--muted)}.idle ul{margin:0;padding:0 14px 14px 32px;columns:2 260px}
.idle li{font-size:.85rem}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.2rem;margin:32px 0 12px}
.muted{color:var(--muted)}.small{font-size:.85rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile b{display:block;font-size:1.6rem;line-height:1.2}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.85rem}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.8rem;font-weight:600}
.PASS{color:var(--pass);background:var(--pass-bg)}.FAIL{color:var(--fail);background:var(--fail-bg)}
.UNVERIFIED{color:var(--unv);background:var(--unv-bg)}.SKIPPED{color:var(--skip);background:var(--skip-bg)}
details{border:1px solid var(--line);border-radius:8px;margin:10px 0;background:var(--bg)}
summary{cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
details>div{padding:0 14px 14px}
code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.85rem}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto;white-space:pre}
.hist span{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:2px}
.hist .PASS{background:var(--pass)}.hist .FAIL{background:var(--fail)}
.hist .UNVERIFIED,.hist .SKIPPED{background:var(--unv)}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _badge(status: str, title: str = "") -> str:
    if not status:
        return '<span class="muted">-</span>'
    t = f' title="{_e(title)}"' if title else ""
    return f'<span class="badge {_e(status)}"{t}>{_e(status)}</span>'


def _summary_tiles(d: dict) -> str:
    c = d["port_counts"]
    frozen = sum(k["reference"]["state"] == "frozen" for k in d["kernels"])
    tiles = [(len(d["kernels"]), "kernels found"), (frozen, "references validated on CUDA"),
             (c["PASS"], "port results PASS"), (c["FAIL"], "FAIL"), (c["UNVERIFIED"], "UNVERIFIED")]
    if c["SKIPPED"]:
        tiles.append((c["SKIPPED"], "SKIPPED"))
    tiles.append((len(d["passed_on"]), "hardware targets with a PASS"))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><b>{n}</b><span class="muted small">{_e(label)}</span></div>'
        for n, label in tiles) + "</div>"


def _cell(k: dict, backend: str) -> str:
    status, tip = cell_status(k, backend)
    if not status:
        return '<td><span class="muted">-</span></td>'
    port = next(p for p in k["ports"] if p["backend"] == backend)
    extra = []
    if status == "PASS" and port["passed_on"]:
        extra.append(_e(", ".join(port["passed_on"])))
    if status == "PASS" and port["speedup"] is not None:
        extra.append(f'<span class="speed">{port["speedup"]:.2f}x vs CUDA</span>')
    sub = "".join(f'<span class="small muted">{x}</span>' for x in extra)
    return f'<td class="cell">{_badge(status, tip)}{sub}</td>'


def _matrix(d: dict) -> str:
    if not d["kernels"]:
        return '<p class="muted">No kernels registered yet. Run scan_kernels.</p>'
    started = [k for k in d["kernels"] if k["started"]]
    if not started:
        return '<p class="muted">No kernel has a reference or a port yet.</p>'
    head = "<tr><th>kernel</th><th>reference</th>" + "".join(f"<th>{_e(b)}</th>" for b in d["backends"]) + "</tr>"
    body = []
    for k in started:
        ref = k["reference"]["state"]
        ref_cell = _badge("PASS", "validated against the original CUDA kernel") if ref == "frozen" \
            else _badge("FAIL" if ref == "tampered" else "SKIPPED") + f' <span class="small muted">{_e(ref)}</span>'
        cells = "".join(_cell(k, b) for b in d["backends"])
        body.append(f'<tr><td><a href="#{_e(_anchor(k))}"><code>{_e(k["name"])}</code></a>'
                    f'<div class="small muted">{_e(k["file"])}</div></td><td>{ref_cell}</td>{cells}</tr>')
    return f'<div class="scroll"><table>{head}{"".join(body)}</table></div>'


def _idle(d: dict) -> str:
    idle = [k for k in d["kernels"] if not k["started"]]
    if not idle:
        return ""
    items = "".join(f'<li><code>{_e(k["name"])}</code> <span class="muted">{_e(k["file"])}:{_e(k["line"])}</span></li>'
                    for k in idle)
    return (f'<details class="idle"><summary>{len(idle)} more kernel{"" if len(idle) == 1 else "s"} '
            f'found by the scan, not started (no reference, no port)</summary><ul>{items}</ul></details>')


def _anchor(k: dict) -> str:
    return "k-" + "".join(ch if ch.isalnum() else "-" for ch in k["id"])


def _bench_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = ["<table><tr><th>hardware</th><th>case</th><th>dtype</th><th>port ms</th>"
           "<th>original ms</th><th>speedup</th><th></th></tr>"]
    for b in rows:
        orig = f"{b['original_ms']:.4f}" if b.get("original_ms") is not None else "-"
        speed = f"{b['speedup']:.2f}x" if b.get("speedup") is not None else "-"
        notes = []
        if not b.get("current"):
            notes.append("stale: port changed since")
        if not b.get("comparable", True):
            notes.append(b.get("note") or "not comparable to GPU timings")
        out.append(f"<tr><td>{_e(b.get('hardware'))}</td><td>{_e(b.get('case'))}</td><td>{_e(b.get('dtype'))}</td>"
                   f"<td class=num>{b.get('port_ms', 0):.4f}</td><td class=num>{orig}</td>"
                   f"<td class=num>{speed}</td><td class='small muted'>{_e('; '.join(notes))}</td></tr>")
    out.append("</table>")
    return "".join(out)


def _port_section(p: dict) -> str:
    rows = "".join(
        f"<tr><td>{_e(r['hardware'])}</td><td>{_badge(r['status'])}</td><td>{_e(r['cases'])}</td>"
        f"<td class='small'>{_e(r['reason'])}</td><td class='small muted'>{_e(r.get('timestamp', ''))}</td></tr>"
        for r in p["rows"])
    a = p["attempts"]
    hist = "".join(f'<span class="{_e(s)}" title="attempt {i}: {_e(s)}"></span>'
                   for i, s in enumerate(a["history"], 1))
    first = f"first PASS on attempt {a['first_pass_attempt']}" if a["first_pass_attempt"] else "never passed"
    tol = ", ".join(f"{dt}: rtol {t['rtol']:g}, atol {t['atol']:g}" for dt, t in sorted(p["tolerances"].items()))
    parts = [f"<h3>{_e(p['backend'])} <span class='small muted'>({p['port_lines']} lines)</span></h3>",
             f"<table><tr><th>hardware</th><th>status</th><th>cases</th><th>reason</th><th>when</th></tr>{rows}</table>",
             f"<p class='small'>Attempts: {a['total']} ({a['failed']} failed), {first} "
             f"<span class='hist'>{hist}</span></p>"]
    if tol:
        parts.append(f"<p class='small muted'>Tolerances used: {_e(tol)}</p>")
    if p["failure_report"]:
        parts.append(f"<p class='small'>Latest failure:</p><pre>{_e(p['failure_report'])}</pre>")
    parts.append(_bench_table(p["bench"]))
    return "".join(parts)


def _kernel_details(k: dict) -> str:
    ref = k["reference"]
    if ref["state"] == "frozen":
        ref_txt = (f"Reference frozen after the original passed {ref['cases_passed']} cases on "
                   f"{_e(ref['validated_on'])} at {_e(ref['validated_at'])} (sha256 {_e(ref['reference_hash'])}...).")
    else:
        ref_txt = f"Reference {_e(ref['state'])}: {_e(ref.get('detail', ''))}"
    statuses = [cell_status(k, p["backend"])[0] for p in k["ports"]]
    badges = " ".join(_badge(s) for s in statuses) or _badge("SKIPPED")
    feats = ", ".join(k["features"]) or "none"
    ports = "".join(_port_section(p) for p in k["ports"]) or "<p class='muted'>No ports written.</p>"
    via = f"<p class='small muted'>Defined by the macro call <code>{_e(k['generated_by'])}</code>.</p>" \
        if k.get("generated_by") else ""
    return (f'<details id="{_e(_anchor(k))}"><summary><code>{_e(k["name"])}</code>{badges}'
            f'<span class="small muted">{_e(k["file"])}:{_e(k["line"])}</span></summary><div>'
            f"<p class='small'>{ref_txt}</p>{via}<p class='small muted'>CUDA features: {_e(feats)}</p>"
            f"{ports}</div></details>")


def _pipelines(d: dict) -> str:
    if not d.get("pipelines"):
        return ""
    rows = "".join(
        f"<tr><td><code>{_e(p['pipeline'])}</code></td><td>{p['stages']}</td><td>{_e(p['backend'])}</td>"
        f"<td>{_e(p['hardware'])}</td><td>{_badge(p['status'], p['reason'])}"
        f"{' <span class=small>' + _e(p['reason']) + '</span>' if p['reason'] else ''}</td><td>{_e(p['cases'])}</td>"
        f"<td class=small>{'<br>'.join(_e(x) for x in p.get('drift', []))}</td></tr>"
        for p in d["pipelines"])
    return ("<h2>Pipelines</h2><p class=small>Kernels run in a chain; every stage is checked against its reference "
            "on what the stage before really wrote. Drift compares the chain with a float64 chain of references; "
            "it is information, not a verdict.</p><table><thead><tr><th>pipeline</th><th>stages</th><th>backend</th>"
            f"<th>hardware</th><th>status</th><th>cases</th><th>drift</th></tr></thead><tbody>{rows}</tbody></table>")


def render(d: dict) -> str:
    """Full HTML page for report data from ``collect``."""
    hw = ", ".join(d["hardware"]) or "none yet"
    data_json = json.dumps(d, default=str).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>kernelport report</title><style>{CSS}</style></head><body><main>
<h1>kernelport report</h1>
<p class="muted small">{_e(d['repo'])} · generated {_e(d['generated'])} · kernelport {_e(d['version'])} · hardware: {_e(hw)}</p>
<p class="headline"><b>{_e(d['headline'])}</b></p>
<p class="small">A port is <b>PASS</b> only if its current code ran on real hardware and matched a reference that was
first validated against the original CUDA kernel. Anything that changed since its last run is shown as UNVERIFIED.</p>
{_summary_tiles(d)}
<h2>Kernels × backends</h2>
{_matrix(d)}
{_idle(d)}
{_pipelines(d)}
<h2>Details</h2>
{''.join(_kernel_details(k) for k in d['kernels'] if k['started']) or '<p class="muted">Nothing started yet.</p>'}
<script type="application/json" id="kernelport-data">{data_json}</script>
</main></body></html>"""


def build_report(ws: Workspace) -> Path:
    """Write .kernelport/report.html and return its path."""
    path = ws.safe_path("report.html")
    path.write_bytes(render(collect(ws)).encode("utf-8"))
    return path
