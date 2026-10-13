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
            f"{b['original_ms']:.4f}" if b else "-",
            f"{b['port_ms']:.4f}" if b else "-",
            f"{b['speedup']:.2f}x" if b else "-",
        )
    con.print(table)
    for e in entries:
        for r in [e.original, *e.ports.values()]:
            if r.status == "FAIL":
                con.print(escape(r.failure_report()))
    failed = any(e.original.status == "FAIL" or any(r.status == "FAIL" for r in e.ports.values())
                 for e in entries)
    raise typer.Exit(1 if failed else 0)


@app.command()
def scan(
    path: str = typer.Argument(".", help="File or directory to scan."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Find CUDA kernels and show their signatures and features."""
    from pathlib import Path

    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport.analyzer.cuda_parser import scan as scan_path

    kernels = scan_path(Path(path))
    if as_json:
        print(json.dumps([k.to_dict(include_source=False) for k in kernels], indent=2))
        return
    table = Table(title=f"{len(kernels)} CUDA kernels in {path}")
    for col in ("id", "line", "params", "features", "launches"):
        table.add_column(col)
    for k in kernels:
        params = ", ".join(f"{p.name}:{p.kind}" for p in k.params)
        table.add_row(escape(k.id), str(k.line), escape(params), ", ".join(k.features), str(len(k.launches)))
    Console().print(table)


if __name__ == "__main__":
    app()
