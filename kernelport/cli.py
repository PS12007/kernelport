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
            (f"{b['speedup']:.2f}x" + ("" if b["comparable"] else " [dim](n/c)[/]"))
            if b and b["speedup"] is not None else "-",
        )
    con.print(table)
    notes = {b["note"] for e in entries for b in e.bench.values() if not b["comparable"]}
    for note in notes:
        con.print(f"[dim](n/c) not comparable: {escape(note)}[/]")
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
def validate(kernel_id: str = typer.Argument(None),
             all_: bool = typer.Option(False, "--all",
                                       help="Every kernel whose candidate reference is not frozen yet."),
             as_json: bool = JSON_OPT) -> None:
    """Run the original CUDA kernel against the reference; freeze on PASS (MCP: validate_reference)."""
    from rich.console import Console
    from rich.markup import escape

    from kernelport import tools
    from kernelport.workspace import Workspace

    if bool(kernel_id) == all_:
        raise typer.BadParameter("give a kernel id or --all (not both)")
    if kernel_id:
        _emit(tools.validate_reference(kernel_id), as_json)
        return
    ws = Workspace()
    ids = ws.pending_validation() if ws.dir.is_dir() else []
    results = []
    for kid in ids:
        r = {"kernel_id": kid, **tools.validate_reference(kid)}
        results.append(r)
        if r.get("status") == "UNAVAILABLE":
            break  # same reason for every kernel
    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        con = Console()
        if not ids:
            con.print("nothing to validate: every candidate reference is already frozen")
        for r in results:
            color = "green" if r.get("ok") else "red"
            con.print(f"[{color}]{escape(r['kernel_id'])}: {escape(r.get('summary', ''))}[/]")
            if r.get("report"):
                con.print(escape(r["report"]))
    if any(not r.get("ok") for r in results):
        raise typer.Exit(1)


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
def tune(kernel_id: str, backend: str, dtype: str = typer.Option("float32"), as_json: bool = JSON_OPT) -> None:
    """Try the port's TUNE combinations, keep the fastest that passes (MCP: tune_port)."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport import tools

    result = tools.tune_port(kernel_id, backend, dtype)
    if not as_json and result.get("rows"):
        table = Table(title=f"tune {kernel_id} / {backend} ({dtype})")
        for col in ("params", "status", "cases", "port ms", ""):
            table.add_column(col)
        for r in result["rows"]:
            table.add_row(escape(json.dumps(r["params"])), r["status"], f"{r['cases_passed']}/{r['cases_total']}",
                          f"{r['port_ms']:.4f}" if r["port_ms"] is not None else "-",
                          "current" if r["current"] else ("best" if r["params"] == result.get("best") else ""))
        Console().print(table)
    _emit(result, as_json)


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
def report(markdown: bool = typer.Option(False, "--markdown", "--md", help="Write report.md and print it."),
           as_json: bool = JSON_OPT) -> None:
    """Build the HTML (or Markdown) report (MCP: make_report)."""
    from kernelport import tools

    res = tools.make_report("markdown" if markdown else "html")
    if markdown and not as_json and res.get("ok"):
        print(res["content"])
        return
    _emit(res, as_json)


TRY_PROMPT = 'port the CUDA kernels in this repo to Triton and verify them'


@app.command()
def setup(
    uninstall: bool = typer.Option(False, "--uninstall", help="Undo everything a previous setup added."),
    print_only: bool = typer.Option(False, "--print", help="Only print manual instructions; change nothing."),
    agent: list[str] = typer.Option(None, "--agent", "-a", help="Only these agents: claude, codex, cursor, gemini."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Apply without asking (still shows the changes)."),
    home: str = typer.Option(None, "--home", hidden=True, help="Home directory override (tests)."),
    no_cli: bool = typer.Option(False, "--no-agent-cli", hidden=True, help="Edit config files instead of `claude mcp add`."),
) -> None:
    """Register the kernelport MCP server and skill with your coding agents."""
    from pathlib import Path

    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from kernelport.setup_agents.detect import agents as all_agents
    from kernelport.setup_agents.detect import server_command
    from kernelport.setup_agents.register import SetupError, apply, plan, plan_uninstall
    from kernelport.setup_agents.register import uninstall as do_uninstall

    con = Console()
    home_p = Path(home) if home else Path.home()

    if uninstall:
        changes = plan_uninstall(home_p)
        if not changes:
            con.print("nothing to uninstall: kernelport setup has not changed any agent config")
            return
        for c in changes:
            con.print(f"- {c.agent}: {escape(c.preview)} ({escape(str(c.path))})")
        if not yes and not typer.confirm("Undo these changes?", default=False):
            raise typer.Exit(1)
        dest = do_uninstall(home_p)
        con.print(f"[green]done.[/] Current files were backed up to {escape(str(dest))}. Restart your agents.")
        return

    found = all_agents(home_p, check_path=not home)
    chosen = [a for a in found if (a.key in agent if agent else a.detected)]
    if agent:
        unknown = set(agent) - {a.key for a in found}
        if unknown:
            con.print(f"[red]unknown agent(s): {', '.join(sorted(unknown))}[/]")
            raise typer.Exit(2)

    table = Table(title="coding agents")
    for col in ("agent", "detected", "MCP config", "skill folder"):
        table.add_column(col)
    for a in found:
        table.add_row(a.label, "yes" if a.detected else "no", escape(str(a.config_path)), escape(str(a.skill_dir)))
    con.print(table)

    command = server_command()
    if print_only:
        con.print("\n[bold]Manual setup[/] (nothing was changed):")
        con.print(f"MCP server command: {escape(' '.join(command))}")
        for c in plan(found, command, use_cli=False):
            con.print(f"\n[bold]{c.agent}[/] {c.kind}: {escape(str(c.path))}\n{escape(c.preview)}")
        return
    if not chosen:
        con.print("No supported agents detected. Use --agent to pick one, or --print for manual steps.")
        raise typer.Exit(1)

    try:
        changes = plan(chosen, command, use_cli=not no_cli)
    except SetupError as exc:
        con.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)
    con.print("\n[bold]Planned changes[/] (every touched file is backed up first):")
    for c in changes:
        con.print(f"\n[bold]{c.agent}[/] {c.kind} [{c.action}] {escape(str(c.path))}"
                  + (" via the agent's own CLI" if c.method == "cli" else ""))
        if c.action != "unchanged":
            con.print(escape(c.preview))
    if all(c.action == "unchanged" for c in changes):
        con.print("\nEverything is already set up.")
        return
    if not yes and not typer.confirm("\nApply these changes?", default=False):
        con.print("No changes made.")
        raise typer.Exit(1)
    try:
        dest = apply(changes, home_p)
    except SetupError as exc:
        con.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)
    con.print(f"\n[green]Done.[/] Backups: {escape(str(dest))}")
    con.print(f'Restart your agent, then try: "{TRY_PROMPT}"')
    con.print("Undo with: kernelport setup --uninstall")


@app.command()
def agent(
    task: str = typer.Argument(TRY_PROMPT, help="What to do, in plain words."),
    model: str = typer.Option(None, help="Claude model id (default: claude-opus-5-5)."),
    effort: str = typer.Option("high", help="low | medium | high | xhigh | max"),
    max_turns: int = typer.Option(80, help="Stop after this many model turns."),
    no_fallbacks: bool = typer.Option(False, "--no-fallbacks", help="Disable server-side refusal fallbacks."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Start without the cost confirmation."),
) -> None:
    """Built-in agent (optional): Claude drives the kernelport tools via YOUR Anthropic API key.

    For people without a coding agent. Needs `pip install kernelport[agent]` and ANTHROPIC_API_KEY
    (or an `ant auth login` profile). Every run is billed to that account.
    """
    from rich.console import Console
    from rich.markup import escape

    from kernelport import agent as kp_agent
    from kernelport.workspace import find_root

    con = Console()
    root = find_root()
    model = model or kp_agent.DEFAULT_MODEL
    con.print(f"repo: {escape(str(root))}\nmodel: {model} (effort {effort}), up to {max_turns} turns\n"
              "This uses the Anthropic API and is billed to your account.")
    if not yes and not typer.confirm("Start?", default=False):
        raise typer.Exit(1)
    try:
        res = kp_agent.run_agent(task, root, model=model, effort=effort, max_turns=max_turns,
                                 fallbacks=not no_fallbacks, on_event=lambda s: con.print(escape(s)))
    except RuntimeError as exc:
        con.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2)
    con.print(f"\n[dim]stop: {res.stop_reason}; {res.turns} turns, {res.tool_calls} tool calls; tokens in "
              f"{res.input_tokens} (cache read {res.cache_read_tokens}), out {res.output_tokens}[/]")
    raise typer.Exit(0 if res.stop_reason == "end_turn" else 1)


@app.command()
def mcp() -> None:
    """Start the MCP server on stdio (used by agent configs)."""
    from kernelport.mcp_server import main as serve

    serve()


if __name__ == "__main__":
    app()
