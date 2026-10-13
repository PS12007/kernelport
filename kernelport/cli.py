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


if __name__ == "__main__":
    app()
