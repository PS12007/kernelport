"""Render a workspace's status table to an SVG for the README.

    python scripts/render_status_svg.py examples/llm.c docs/img/llmc-status.svg
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelport import tools  # noqa: E402

COLORS = {"PASS": "bold green", "FAIL": "bold red", "UNVERIFIED": "yellow", "SKIPPED": "dim"}
BACKEND_LABEL = {"triton": "Triton", "metal": "Metal (MLX)", "webgpu": "WebGPU (WGSL)", "cpu": "CPU", "hip": "HIP"}


def main(workspace: str, out: str) -> None:
    res = tools.list_status(root=workspace)
    rows = [r for r in res["rows"] if r["backend"] != "-"]
    table = Table(title="kernelport status: llm.c dev/cuda kernels, unmodified CUDA files", title_style="bold")
    table.add_column("kernel", no_wrap=True)
    table.add_column("port")
    table.add_column("verified on", no_wrap=True)
    table.add_column("status")
    table.add_column("cases", justify="right")
    for r in rows:
        hw = r["hardware"].replace(" Laptop GPU", "").replace("NVIDIA GeForce ", "")
        table.add_row(r["kernel"].split("::")[-1], BACKEND_LABEL.get(r["backend"], r["backend"]), hw,
                      f"[{COLORS[r['status']]}]{r['status']}[/]", r["cases"])
    con = Console(record=True, width=118, file=open("NUL" if sys.platform == "win32" else "/dev/null", "w"))
    con.print(table)
    con.print("[dim]Every reference was validated against the original CUDA kernel on the RTX 5060 and "
              "frozen;\nevery port was checked against that frozen reference on the hardware shown.[/]")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    con.save_svg(out, title="kernelport")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
