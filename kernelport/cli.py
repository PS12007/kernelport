"""kernelport command line interface."""

from __future__ import annotations

import json

import typer

app = typer.Typer(help="Port CUDA kernels to other backends and prove they are correct.",
                  no_args_is_help=True)


@app.callback()
def main() -> None:
    """kernelport: your agent writes the port, kernelport runs it and proves it."""


@app.command()
def doctor(as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table.")) -> None:
    """Show environment info and smoke-test every available backend."""
    from kernelport.doctor import print_report, run_doctor

    report = run_doctor()
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    raise typer.Exit(0 if report.ok else 1)


@app.command()
def zoo(
    backend: str = typer.Option("triton", help="Port backend to verify."),
    kernel: list[str] = typer.Option(None, "--kernel", "-k", help="Only these kernels."),
    no_bench: bool = typer.Option(False, "--no-bench", help="Skip benchmarking."),
) -> None:
    """Validate zoo originals, verify stored ports, and benchmark passing ports."""
    from pathlib import Path

    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport.zoo import find_zoo, run_zoo

    con = Console()
    entries = run_zoo(find_zoo(), kernel or None, (backend,), bench=not no_bench,
                      out_dir=Path(".kernelport") / "zoo")
    colors = {"PASS": "green", "FAIL": "red", "UNVERIFIED": "yellow", "SKIPPED": "dim"}
    table = Table(title=f"zoo: original vs reference, {backend} port vs reference")
    for col in ("kernel", "original", "cases", backend, "cases", "orig ms", "port ms", "speedup"):
        table.add_column(col)
    for e in entries:
        p = e.ports[backend]
        b = e.bench.get(backend)
        table.add_row(
            e.kernel,
            f"[{colors[e.original.status]}]{e.original.status}[/]",
            f"{e.original.num_passed}/{len(e.original.cases)}",
            f"[{colors[p.status]}]{p.status}[/]",
            f"{p.num_passed}/{len(p.cases)}" if p.cases else escape(p.reason),
            f"{b['original_ms']:.4f}" if b and b["original_ms"] is not None else "-",
            f"{b['port_ms']:.4f}" if b else "-",
            f"{b['speedup']:.2f}x" if b and b["speedup"] is not None else "-",
        )
    con.print(table)
    for e in entries:
        for r in [e.original, *e.ports.values()]:
            if r.status == "FAIL":
                con.print(escape(r.failure_report()))
    failed = any(e.original.status == "FAIL" or any(r.status == "FAIL" for r in e.ports.values())
                 for e in entries)
    raise typer.Exit(1 if failed else 0)


def _emit(result: dict, as_json: bool, extra_keys: tuple[str, ...] = ("report",)) -> None:
    """Print a tool result: JSON, or summary plus any report text."""
    from rich.console import Console
    from rich.markup import escape

    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        con = Console()
        color = "green" if result.get("ok") else "red"
        con.print(f"[{color}]{escape(result.get('summary', ''))}[/]")
        for k in extra_keys:
            if result.get(k):
                con.print(escape(str(result[k])))
    if not result.get("ok", False):
        raise typer.Exit(1)


JSON_OPT = typer.Option(False, "--json", help="Print the full JSON result.")


@app.command()
def scan(path: str = typer.Argument(".", help="File or directory inside this repo."), as_json: bool = JSON_OPT) -> None:
    """Find CUDA kernels and register them in .kernelport/ (MCP: scan_kernels)."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport import tools

    res = tools.scan_kernels(path)
    if as_json or not res.get("ok"):
        _emit(res, as_json)
        return
    table = Table(title=res["summary"])
    for col in ("kernel id", "line", "params", "features"):
        table.add_column(col)
    for k in res["kernels"]:
        table.add_row(escape(k["kernel_id"]), str(k["line"]), escape(", ".join(k["params"])), ", ".join(k["features"]))
    Console().print(table)


@app.command()
def read(kernel_id: str, as_json: bool = JSON_OPT) -> None:
    """Show a kernel's source, signature and input spec (MCP: read_kernel)."""
    from kernelport import tools

    res = tools.read_kernel(kernel_id)
    if not as_json and res.get("ok"):
        print(res["source"])
        print(json.dumps(res["input_spec"], indent=2))
    _emit(res, as_json, ())


@app.command()
def skill(name: str = typer.Argument("workflow")) -> None:
    """Print a skill file (MCP: read_skill)."""
    from kernelport import tools

    res = tools.read_skill(name)
    if res.get("ok"):
        print(res["content"])
    else:
        _emit(res, False)


@app.command("write-reference")
def write_reference_cmd(kernel_id: str, file: str, spec: str = typer.Option(None, help="launch.json to save too."),
                        as_json: bool = JSON_OPT) -> None:
    """Save a candidate reference from a file (MCP: write_reference)."""
    from pathlib import Path

    from kernelport import tools

    spec_obj = json.loads(Path(spec).read_text(encoding="utf-8")) if spec else None
    _emit(tools.write_reference(kernel_id, Path(file).read_text(encoding="utf-8"), spec_obj), as_json)


@app.command()
def validate(kernel_id: str, as_json: bool = JSON_OPT) -> None:
    """Run the original CUDA kernel against the reference; freeze on PASS (MCP: validate_reference)."""
    from kernelport import tools

    _emit(tools.validate_reference(kernel_id), as_json)


@app.command("write-port")
def write_port_cmd(kernel_id: str, backend: str, file: str, as_json: bool = JSON_OPT) -> None:
    """Save a port from a file (MCP: write_port)."""
    from pathlib import Path

    from kernelport import tools

    _emit(tools.write_port(kernel_id, backend, Path(file).read_text(encoding="utf-8")), as_json)


def _saved_ports(backend: str | None, kernel: str | None) -> list[tuple[str, str]]:
    from kernelport.backends.base import BACKEND_NAMES
    from kernelport.workspace import Workspace

    ws = Workspace()
    pairs = []
    ids = [ws.get(kernel).id] if kernel else ws.kernel_ids() if ws.dir.is_dir() else []
    for kid in ids:
        for b in ([backend] if backend else BACKEND_NAMES):
            if ws.port_path(kid, b).is_file():
                pairs.append((kid, b))
    return pairs


@app.command()
def verify(backend: str = typer.Option(None, help="Only this backend."),
           kernel: str = typer.Option(None, help="Only this kernel id."),
           as_json: bool = JSON_OPT) -> None:
    """Verify saved ports in .kernelport/ on this machine (MCP: verify_port)."""
    from rich.console import Console
    from rich.markup import escape

    from kernelport import tools

    pairs = _saved_ports(backend, kernel)
    results = [{"kernel_id": k, "backend": b, **tools.verify_port(k, b)} for k, b in pairs]
    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        con = Console()
        if not pairs:
            con.print("no saved ports to verify")
        for r in results:
            color = {"PASS": "green", "FAIL": "red"}.get(r.get("status"), "yellow")
            con.print(f"[{color}]{escape(r['kernel_id'])} / {r['backend']}: {escape(r['summary'])}[/]")
            if r.get("report"):
                con.print(escape(r["report"]))
    if any(r.get("status") == "FAIL" or r.get("status") == "REFUSED" for r in results):
        raise typer.Exit(1)


@app.command()
def bench(backend: str = typer.Option(None, help="Only this backend."),
          kernel: str = typer.Option(None, help="Only this kernel id."),
          dtype: str = typer.Option("float32"), as_json: bool = JSON_OPT) -> None:
    """Benchmark saved ports that passed on this machine (MCP: bench_port)."""
    from rich.console import Console
    from rich.markup import escape

    from kernelport import tools

    results = [{"kernel_id": k, "backend": b, **tools.bench_port(k, b, {"dtype": dtype})}
               for k, b in _saved_ports(backend, kernel)]
    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return
    for r in results:
        Console().print(f"{escape(r['kernel_id'])} / {r['backend']}: {escape(r['summary'])}")


@app.command()
def status(as_json: bool = JSON_OPT) -> None:
    """Kernel x backend x hardware status table (MCP: list_status)."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport import tools

    res = tools.list_status()
    if as_json:
        print(json.dumps(res, indent=2))
        return
    colors = {"PASS": "green", "FAIL": "red", "UNVERIFIED": "yellow", "SKIPPED": "dim"}
    table = Table(title=res["summary"])
    for col in ("kernel", "reference", "backend", "hardware", "status", "cases", "reason"):
        table.add_column(col)
    for r in res["rows"]:
        table.add_row(escape(r["kernel"]), r["reference"], r["backend"], escape(r["hardware"]),
                      f"[{colors[r['status']]}]{r['status']}[/]", r["cases"], escape(r["reason"]))
    Console().print(table)


@app.command()
def report(as_json: bool = JSON_OPT) -> None:
    """Build the HTML report (MCP: make_report)."""
    from kernelport import tools

    _emit(tools.make_report(), as_json)


@app.command()
def mcp() -> None:
    """Start the MCP server on stdio (used by agent configs)."""
    from kernelport.mcp_server import main as serve

    serve()


if __name__ == "__main__":
    app()
