"""Record a real kernelport session and render it as an animated SVG terminal for the README.

    python scripts/record_demo.py record   # run the CLI on this machine (needs CUDA + Triton)
    python scripts/record_demo.py render   # docs/demo/transcript.json -> docs/img/demo.svg (no GPU)
    python scripts/record_demo.py          # both

The session ports the zoo's softmax kernel to Triton in a throwaway repo: scan, validate the
reference against the original CUDA kernel, verify a first port that forgets the max
subtraction (FAIL), verify the fixed port (PASS), bench, status. The ports are the ones an agent
would write (tests/bad_ports/softmax_no_max.py, zoo/softmax/ports/triton.py). Every output line
in the SVG is exactly what the CLI printed; the renderer only adds colour.
"""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "docs" / "demo" / "transcript.json"
SVG = ROOT / "docs" / "img" / "demo.svg"
COLS = 110

# (comment, command, file to place before running it: (name in repo, source in this checkout))
FRAMES = [
    [("# a repo with one CUDA kernel", "kernelport scan .", None)],
    [("# the agent writes a PyTorch reference and an input spec ...", "kernelport write-reference softmax "
      "reference.py --spec launch.json", None),
     ("# ... which must match the ORIGINAL CUDA kernel before anything else", "kernelport validate softmax", None)],
    [("# the agent's first Triton port", "kernelport write-port softmax triton port.py",
      ("port.py", "tests/bad_ports/softmax_no_max.py")),
     ("", "kernelport verify softmax", None)],
    [("# the agent reads the report and adds the max subtraction", "kernelport write-port softmax triton port.py",
      ("port.py", "zoo/softmax/ports/triton.py")),
     ("", "kernelport verify softmax", None),
     ("", "kernelport bench softmax", None)],
    [("", "kernelport status", None)],
]


# --------------------------------------------------------------------------- record

def record() -> None:
    sys.path.insert(0, str(ROOT))
    from kernelport import __version__

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / ".git").mkdir(parents=True)
        shutil.copy(ROOT / "zoo" / "softmax" / "kernel.cu", repo / "softmax.cu")
        spec = json.loads((ROOT / "zoo" / "softmax" / "launch.json").read_text(encoding="utf-8"))
        spec["source"] = "softmax.cu"
        (repo / "launch.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        shutil.copy(ROOT / "zoo" / "softmax" / "reference.py", repo / "reference.py")
        env = {**os.environ, "COLUMNS": str(COLS), "PYTHONIOENCODING": "utf-8"}
        env.pop("KERNELPORT_ROOT", None)
        frames = []
        for frame in FRAMES:
            steps = []
            for comment, cmd, place in frame:
                if place:
                    shutil.copy(ROOT / place[1], repo / place[0])
                args = cmd.split()[1:]
                p = subprocess.run([sys.executable, "-m", "kernelport", *args], cwd=repo, env=env,
                                   capture_output=True, text=True, encoding="utf-8")
                out = (p.stdout + p.stderr).replace("\r", "")
                lines = [ln.rstrip() for ln in out.rstrip("\n").split("\n")]
                steps.append({"comment": comment, "cmd": cmd, "output": lines, "exit": p.returncode})
                print(f"$ {cmd}  [exit {p.returncode}]\n" + "\n".join(lines) + "\n")
            frames.append(steps)
    TRANSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT.write_text(json.dumps({
        "recorded": datetime.date.today().isoformat(), "kernelport": __version__, "columns": COLS,
        "frames": frames}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TRANSCRIPT.relative_to(ROOT)}")


# --------------------------------------------------------------------------- render

FONT = 13
CW = FONT * 0.6  # monospace advance
LH = 18
PAD = 16
TOP = 28
TYPE_S = 0.035  # seconds per typed character
COLORS = {"fg": "#e6edf3", "dim": "#8b949e", "pass": "#3fb950", "fail": "#f85149", "warn": "#d29922",
          "prompt": "#58a6ff", "bg": "#0d1117", "chrome": "#161b22", "line": "#30363d"}
_TOKENS = re.compile(r"\bPASS\b|\bFAIL\b|\bUNVERIFIED\b|'nan'|unexpected NaN|\bReference frozen\b")


def _spans(line: str) -> str:
    out, last = [], 0
    for m in _TOKENS.finditer(line):
        out.append(html.escape(line[last:m.start()]))
        t = m.group(0)
        color = COLORS["pass"] if t in ("PASS", "Reference frozen") else \
            COLORS["warn"] if t == "UNVERIFIED" else COLORS["fail"]
        out.append(f'<tspan fill="{color}" font-weight="bold">{html.escape(t)}</tspan>')
        last = m.end()
    out.append(html.escape(line[last:]))
    return "".join(out)


# box-drawing characters -> the cell edges they connect (up, down, left, right); drawn as lines so
# tables join up whatever font the viewer has
_BOX = {"─": "lr", "│": "ud", "┌": "rd", "┐": "ld", "└": "ur", "┘": "ul", "├": "udr", "┤": "udl",
        "┬": "lrd", "┴": "lru", "┼": "udlr"}


def _box_path(line: str, x0: float, top: float) -> tuple[str, str]:
    """(line with box characters blanked, SVG path drawing them) for a row whose cell starts at ``top``."""
    d = []
    for i, ch in enumerate(line):
        dirs = _BOX.get(ch)
        if not dirs:
            continue
        cx, cy = x0 + (i + 0.5) * CW, top + LH / 2
        ends = {"u": (cx, top), "d": (cx, top + LH), "l": (x0 + i * CW, cy), "r": (x0 + (i + 1) * CW, cy)}
        d += [f"M{cx:.1f} {cy:.1f}L{ends[s][0]:.1f} {ends[s][1]:.1f}" for s in dirs]
    return "".join(" " if c in _BOX else c for c in line), "".join(d)


def _wrap(line: str, cols: int) -> list[str]:
    return [line[i:i + cols] for i in range(0, len(line), cols)] or [""]


def render(out: Path = SVG, still: int | None = None) -> None:
    """Write the animated SVG, or with ``still`` one frame without animation (to check the layout)."""
    t = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    cols = t["columns"]
    # lay out each frame: (kind, text, appear_time) with kind in comment / cmd / out
    frames, now = [], 0.6
    for steps in t["frames"]:
        start, rows = now, []
        for s in steps:
            if s["comment"]:
                rows.append(("comment", s["comment"], now))
            rows.append(("cmd", s["cmd"], now))
            now += len(s["cmd"]) * TYPE_S + 0.5
            for ln in s["output"]:
                for part in _wrap(ln, cols):
                    rows.append(("out", part, now))
            now += 1.2 + min(len(s["output"]), 20) * 0.12
        now += 1.0
        frames.append((start, now, rows))
        now += 0.15
    total = now + 2.5  # hold the last frame
    frames[-1] = (frames[-1][0], total, frames[-1][2])
    n_rows = max(len(r) for _, _, r in frames)
    width = int(PAD * 2 + cols * CW)
    height = int(TOP + PAD + n_rows * LH + PAD)

    css, body = [], []

    def pct(x: float) -> str:
        return f"{100 * x / total:.3f}%"

    def show(name: str, a: float, b: float) -> None:
        eps = 0.01
        css.append(f"@keyframes {name}{{0%,{pct(max(a - eps, 0))}{{opacity:0}}{pct(a)},{pct(b - eps)}{{opacity:1}}"
                   f"{pct(b)},100%{{opacity:0}}}}.{name}{{opacity:0;animation:{name} {total:.2f}s linear infinite}}")

    k = 0
    for fi, (_, fb, rows) in enumerate(frames):
        if still is not None and fi != still:
            continue
        for ri, (kind, text, at) in enumerate(rows):
            k += 1
            name = f"r{k}"
            if still is None:
                show(name, at, fb)
            y = TOP + PAD + ri * LH + FONT

            def fit(n: int) -> str:  # pin the text to the character grid whatever monospace font is used
                return f' textLength="{n * CW:.1f}" lengthAdjust="spacing"' if n else ""

            if kind == "cmd":
                n = len(text) + 2
                body.append(f'<g class="{name}"><text x="{PAD}" y="{y}"{fit(n)}>'
                            f'<tspan fill="{COLORS["prompt"]}">$ </tspan>'
                            f'<tspan fill="{COLORS["fg"]}" font-weight="bold">{html.escape(text)}</tspan></text>'
                            + ("" if still is not None else
                               f'<rect class="t{k}" x="{PAD + 2 * CW:.1f}" y="{y - FONT}" '
                               f'width="{(n - 2) * CW + 2:.1f}" height="{LH}" fill="{COLORS["bg"]}"/>')
                            + "</g>")
                dur = (n - 2) * TYPE_S
                css.append(f"@keyframes t{k}{{0%,{pct(at)}{{transform:translateX(0)}}"
                           f"{pct(at + dur)},100%{{transform:translateX({(n - 2) * CW + 2:.1f}px)}}}}"
                           f".t{k}{{animation:t{k} {total:.2f}s steps({n - 2}) infinite}}")
            elif kind == "comment":
                body.append(f'<text class="{name}" x="{PAD}" y="{y}" fill="{COLORS["dim"]}"{fit(len(text))} '
                            f'font-style="italic">{html.escape(text)}</text>')
            else:
                text, path = _box_path(text, PAD, y - FONT)
                lines = f'<path d="{path}" stroke="{COLORS["dim"]}" fill="none"/>' if path else ""
                body.append(f'<g class="{name}"><text x="{PAD}" y="{y}" fill="{COLORS["fg"]}"{fit(len(text))}>'
                            f'{_spans(text)}</text>{lines}</g>')
    title = f"kernelport {html.escape(t['kernelport'])}: CUDA softmax to Triton, recorded {html.escape(t['recorded'])}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>{title}</title>
<style>
text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;font-size:{FONT}px;white-space:pre}}
{chr(10).join(css)}
@media (prefers-reduced-motion: reduce){{[class]{{animation:none!important}}}}
</style>
<rect width="{width}" height="{height}" rx="8" fill="{COLORS["bg"]}" stroke="{COLORS["line"]}"/>
<rect width="{width}" height="28" rx="8" fill="{COLORS["chrome"]}"/>
<rect y="20" width="{width}" height="8" fill="{COLORS["chrome"]}"/>
<circle cx="18" cy="14" r="5" fill="#ff5f57"/><circle cx="36" cy="14" r="5" fill="#febc2e"/>
<circle cx="54" cy="14" r="5" fill="#28c840"/>
<text x="{width / 2:.0f}" y="18" fill="{COLORS["dim"]}" text-anchor="middle" font-size="12">{title}</text>
<g xml:space="preserve">
{chr(10).join(body)}
</g>
</svg>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({width}x{height}, {total:.1f} s loop, {k} rows)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("record", "all"):
        record()
    if what in ("render", "all"):
        render()
    if what == "still":  # python scripts/record_demo.py still <frame> <out.svg>
        render(Path(sys.argv[3]), int(sys.argv[2]))
