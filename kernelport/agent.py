"""Optional built-in agent: drives the kernelport tools with Claude through the Anthropic API.

For people who do not use a coding agent. It needs `pip install kernelport[agent]` and the
user's own Anthropic credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile); every
run is billed to that account. The agent can only act through the 12 kernelport tools, so
all writes stay inside the repo's .kernelport/ workspace and it cannot run shell commands.

The loop is written by hand (not the SDK's beta tool runner) so the turn budget, the tool
surface and the client are fully under kernelport's control and testable with a fake client.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kernelport import tools as kp_tools
from kernelport.mcp_server import INSTRUCTIONS

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TURNS = 80
MAX_TOKENS = 16000
MAX_RESULT_CHARS = 60_000

SYSTEM_PREAMBLE = """You are kernelport's built-in agent. You port CUDA kernels to other backends and prove
each port correct, using only the kernelport tools. Follow the workflow skill below exactly.
Work autonomously: do not ask the user questions; make reasonable choices and state them.
When finished, reply with the markdown report from make_report(format="markdown") and a short
summary of anything that is not PASS and why. Never describe a port as working unless
verify_port returned PASS."""


def _json_type(annotation: Any) -> dict:
    """JSON schema for a tool parameter annotation (str, dict, and their Optional forms)."""
    if isinstance(annotation, str):  # from __future__ annotations
        annotation = eval(annotation, {"dict": dict, "str": str, "Any": Any})  # noqa: S307 (own source)
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if args else {}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def tool_definitions() -> list[dict]:
    """Anthropic tool definitions for every kernelport tool (deterministic order, no `root`)."""
    defs = []
    for fn in kp_tools.TOOLS:
        props, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            if name == "root":
                continue  # the workspace is fixed by the agent, never chosen by the model
            props[name] = _json_type(p.annotation)
            if p.default is inspect.Parameter.empty:
                required.append(name)
        doc = inspect.getdoc(fn) or fn.__name__
        defs.append({"name": fn.__name__, "description": doc,
                     "input_schema": {"type": "object", "properties": props, "required": required}})
    return defs


def system_prompt() -> str:
    """Stable system prompt (cached): preamble + MCP instructions + the workflow skill."""
    skill = kp_tools.read_skill("workflow").get("content", "")
    return f"{SYSTEM_PREAMBLE}\n\n{INSTRUCTIONS}\n\n<workflow_skill>\n{skill}\n</workflow_skill>"


def run_tool(name: str, args: dict, root: Path) -> tuple[str, bool, str]:
    """Execute one tool call. Returns (result JSON text, is_error, one-line summary)."""
    fns = {fn.__name__: fn for fn in kp_tools.TOOLS}
    if name not in fns:
        return json.dumps({"error": f"unknown tool {name}"}), True, f"unknown tool {name}"
    fn = fns[name]
    accepted = set(inspect.signature(fn).parameters) - {"root"}
    bad = set(args) - accepted
    if bad:
        msg = f"unknown arguments {sorted(bad)} for {name}; accepted: {sorted(accepted)}"
        return json.dumps({"error": msg}), True, msg
    kwargs = dict(args)
    if "root" in inspect.signature(fn).parameters:
        kwargs["root"] = str(root)
    try:
        result = fn(**kwargs)
    except Exception as exc:  # a tool bug must not kill the loop; the model sees the error
        msg = f"{type(exc).__name__}: {exc}"
        return json.dumps({"error": msg}), True, msg
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + ' ...[truncated]"}'
    return text, not result.get("ok", True) and "error" in result, str(result.get("summary", ""))


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    final_text: str
    stop_reason: str
    turns: int
    tool_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    log: list[str] = field(default_factory=list)


def make_client():
    """The real Anthropic client (imported only here, so kernelport works without the SDK)."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("the agent needs the Anthropic SDK: pip install kernelport[agent]") from exc
    return anthropic.Anthropic()


def run_agent(task: str, root: Path, *, client=None, model: str = DEFAULT_MODEL,
              effort: str = DEFAULT_EFFORT, max_turns: int = DEFAULT_MAX_TURNS,
              fallbacks: bool = True, on_event: Callable[[str], None] = print) -> AgentResult:
    """Run the tool-use loop until Claude finishes, refuses, or the turn budget is spent."""
    client = client or make_client()
    tools = tool_definitions()
    system = [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}]
    messages: list[dict] = [{"role": "user", "content": task}]
    res = AgentResult(final_text="", stop_reason="", turns=0, tool_calls=0)

    def emit(line: str) -> None:
        res.log.append(line)
        on_event(line)

    extra: dict[str, Any] = {}
    if fallbacks:  # on a safety decline, the API retries on a fallback model in the same call
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    while res.turns < max_turns:
        res.turns += 1
        response = client.beta.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=system, tools=tools, messages=messages,
            output_config={"effort": effort}, cache_control={"type": "ephemeral"}, **extra)
        usage = getattr(response, "usage", None)
        if usage is not None:
            res.input_tokens += getattr(usage, "input_tokens", 0) or 0
            res.output_tokens += getattr(usage, "output_tokens", 0) or 0
            res.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        res.stop_reason = response.stop_reason
        for block in response.content:
            if block.type == "text" and block.text.strip():
                res.final_text = block.text
                emit(block.text)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            emit("Claude declined to continue this request (stop_reason: refusal).")
            return res
        if response.stop_reason == "max_tokens":
            emit("response hit max_tokens; stopping.")
            return res
        calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not calls:
            return res
        results = []
        for call in calls:
            res.tool_calls += 1
            text, is_error, summary = run_tool(call.name, dict(call.input or {}), root)
            emit(f"  [{call.name}] {summary}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text,
                            **({"is_error": True} if is_error else {})})
        messages.append({"role": "user", "content": results})  # all results in one message

    res.stop_reason = "max_turns"
    emit(f"turn budget of {max_turns} used up; stopping. Run `kernelport status` to see where it got to.")
    return res
