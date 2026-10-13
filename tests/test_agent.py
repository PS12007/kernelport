"""Built-in agent loop, driven by a scripted fake client: no network, no SDK, no API credits."""

import json
import shutil
from types import SimpleNamespace as NS

import pytest

from conftest import ZOO
from kernelport import agent, tools


def _text(t):
    return NS(type="text", text=t)


def _call(i, name, **inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: vars(o))))
        stop, content = self.script.pop(0)
        return NS(stop_reason=stop, content=content,
                  usage=NS(input_tokens=100, output_tokens=10, cache_read_input_tokens=90))


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    shutil.copytree(ZOO / "relu", r / "kernels", ignore=shutil.ignore_patterns("ports", "__pycache__"))
    (r / ".git").mkdir()
    return r


def test_tool_definitions_cover_every_tool_without_root():
    defs = {d["name"]: d for d in agent.tool_definitions()}
    assert set(defs) == {fn.__name__ for fn in tools.TOOLS}
    for d in defs.values():
        assert "root" not in d["input_schema"]["properties"]
        assert d["description"]
    assert defs["write_port"]["input_schema"]["required"] == ["kernel_id", "backend", "code"]
    assert defs["write_reference"]["input_schema"]["properties"]["spec"] == {"type": "object"}
    assert defs["bench_port"]["input_schema"]["properties"]["config"] == {"type": "object"}
    assert defs["kernelport_doctor"]["input_schema"]["required"] == []


def test_system_prompt_is_stable_and_contains_the_skill():
    a, b = agent.system_prompt(), agent.system_prompt()
    assert a == b  # no timestamps: keeps the prompt cache valid
    assert "verify_port" in a and "<workflow_skill>" in a


def test_loop_runs_tools_and_returns_final_text(repo):
    client = FakeClient([
        ("tool_use", [_text("Scanning."), _call(1, "scan_kernels", path=".")]),
        ("tool_use", [_call(2, "read_kernel", kernel_id="relu"), _call(3, "list_status")]),
        ("end_turn", [_text("Done: nothing verified yet.")]),
    ])
    events = []
    res = agent.run_agent("port it", repo, client=client, on_event=events.append)
    assert res.stop_reason == "end_turn" and res.turns == 3 and res.tool_calls == 3
    assert res.final_text == "Done: nothing verified yet."
    assert res.input_tokens == 300 and res.cache_read_tokens == 270
    assert any("found 1 kernels" in e for e in events)

    # the tool really ran against this repo's workspace
    assert (repo / ".kernelport" / "index.json").is_file()
    # both parallel results went back in ONE user message, matched by id
    second = client.requests[2]["messages"]
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_2", "tu_3"]
    # request shape: default model, caching, effort, fallbacks, deterministic tools
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"] == {"effort": "high"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert [t["name"] for t in req["tools"]] == [t["name"] for t in client.requests[1]["tools"]]


def test_tool_errors_are_reported_not_raised(repo):
    client = FakeClient([
        ("tool_use", [_call(1, "verify_port", kernel_id="nope", backend="triton"),
                      _call(2, "write_port", kernel_id="relu", backend="triton", code="x", root="/etc"),
                      _call(3, "rm_rf")]),
        ("end_turn", [_text("ok")]),
    ])
    agent.run_agent("x", repo, client=client, on_event=lambda s: None)
    results = client.requests[1]["messages"][-1]["content"]
    assert all(r.get("is_error") for r in results)
    assert "unknown arguments ['root']" in results[1]["content"]  # the model cannot pick the workspace
    assert "unknown tool rm_rf" in results[2]["content"]


def test_turn_budget_and_refusal_stop_the_loop(repo):
    looping = FakeClient([("tool_use", [_call(i, "list_status")]) for i in range(5)])
    res = agent.run_agent("x", repo, client=looping, max_turns=3, on_event=lambda s: None)
    assert res.stop_reason == "max_turns" and res.turns == 3

    refused = FakeClient([("refusal", [])])
    res = agent.run_agent("x", repo, client=refused, on_event=lambda s: None)
    assert res.stop_reason == "refusal" and res.turns == 1


def test_no_fallbacks_option(repo):
    client = FakeClient([("end_turn", [_text("hi")])])
    agent.run_agent("x", repo, client=client, fallbacks=False, on_event=lambda s: None)
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


def test_missing_sdk_gives_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"kernelport\[agent\]"):
        agent.make_client()
