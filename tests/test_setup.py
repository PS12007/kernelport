"""kernelport setup: plan, backup, apply, uninstall. Uses a fake home dir only."""

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kernelport.cli import app
from kernelport.setup_agents.backup import load_state
from kernelport.setup_agents.detect import agents, server_command
from kernelport.setup_agents.register import SetupError, apply, plan, uninstall

CMD = ["/opt/py/bin/python", "-m", "kernelport", "mcp"]


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    (h / ".codex").mkdir()
    (h / ".cursor").mkdir()
    (h / ".gemini").mkdir()
    return h


def _agents(home):
    return agents(home, check_path=False)


def _snapshot(home):
    return {p: p.read_bytes() for p in home.rglob("*") if p.is_file() and ".kernelport" not in p.parts}


def test_detection(home, tmp_path):
    assert {a.key for a in _agents(home) if a.detected} == {"claude", "codex", "cursor", "gemini"}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not any(a.detected for a in agents(empty, check_path=False))


def test_plan_writes_nothing(home):
    before = _snapshot(home)
    plan(_agents(home), CMD, use_cli=False)
    assert _snapshot(home) == before


def test_install_all_agents_and_uninstall_restores_exactly(home):
    # Pre-existing user config that must survive untouched.
    (home / ".claude.json").write_text(json.dumps({"numStartups": 3, "mcpServers": {"other": {"command": "x"}}}))
    (home / ".codex" / "config.toml").write_text('model = "o3"\n\n[mcp_servers.other]\ncommand = "y"\n')
    (home / ".gemini" / "settings.json").write_text(json.dumps({"theme": "dark"}))
    before = _snapshot(home)

    dest = apply(plan(_agents(home), CMD, use_cli=False), home)
    assert (dest / "manifest.json").is_file()

    claude = json.loads((home / ".claude.json").read_text())
    assert claude["numStartups"] == 3 and claude["mcpServers"]["other"] == {"command": "x"}
    assert claude["mcpServers"]["kernelport"] == {"type": "stdio", "command": CMD[0], "args": CMD[1:]}

    codex = tomllib.loads((home / ".codex" / "config.toml").read_text())
    assert codex["model"] == "o3" and codex["mcp_servers"]["other"]["command"] == "y"
    assert codex["mcp_servers"]["kernelport"] == {"command": CMD[0], "args": CMD[1:]}

    cursor = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert cursor["mcpServers"]["kernelport"]["env"] == {"KERNELPORT_ROOT": "${workspaceFolder}"}

    gemini = json.loads((home / ".gemini" / "settings.json").read_text())
    assert gemini["theme"] == "dark" and "type" not in gemini["mcpServers"]["kernelport"]

    for skill in (home / ".claude/skills/kernelport", home / ".agents/skills/kernelport",
                  home / ".cursor/skills/kernelport", home / ".gemini/skills/kernelport"):
        if skill.parent.parent.name == ".agents":
            continue  # codex skill dir is ~/.agents/skills
        assert (skill / "SKILL.md").is_file() and (skill / "backends" / "triton.md").is_file()
    assert (home / ".agents/skills/kernelport/SKILL.md").is_file()

    uninstall(home)
    assert _snapshot(home) == before, "uninstall must restore every file byte-for-byte"
    assert not (home / ".cursor" / "mcp.json").exists()  # created by setup, so removed
    assert load_state(home) == {}


def test_rerun_is_unchanged(home):
    apply(plan(_agents(home), CMD, use_cli=False), home)
    again = plan(_agents(home), CMD, use_cli=False)
    assert {c.action for c in again if c.kind == "mcp"} == {"unchanged"}


def test_previous_kernelport_entry_is_restored(home):
    old = {"command": "old-kernelport", "args": []}
    (home / ".gemini" / "settings.json").write_text(json.dumps({"mcpServers": {"kernelport": old}}))
    gem = [a for a in _agents(home) if a.key == "gemini"]
    changes = plan(gem, CMD, use_cli=False)
    assert changes[0].action == "update"
    apply(changes, home)
    uninstall(home)
    assert json.loads((home / ".gemini" / "settings.json").read_text())["mcpServers"]["kernelport"] == old


def test_invalid_json_refused(home):
    (home / ".claude.json").write_text("{ not json")
    with pytest.raises(SetupError, match="not valid JSON"):
        plan([a for a in _agents(home) if a.key == "claude"], CMD, use_cli=False)


def test_foreign_toml_table_refused(home):
    (home / ".codex" / "config.toml").write_text('[mcp_servers.kernelport]\ncommand = "mine"\n')
    with pytest.raises(SetupError, match="not added by kernelport"):
        plan([a for a in _agents(home) if a.key == "codex"], CMD, use_cli=False)


def test_windows_paths_survive_toml(home):
    cmd = [r"C:\Users\me\venv\Scripts\python.exe", "-m", "kernelport", "mcp"]
    apply(plan([a for a in _agents(home) if a.key == "codex"], cmd, use_cli=False), home)
    parsed = tomllib.loads((home / ".codex" / "config.toml").read_text())
    assert parsed["mcp_servers"]["kernelport"]["command"] == cmd[0]


def test_existing_skill_dir_backed_up_and_restored(home):
    mine = home / ".claude" / "skills" / "kernelport"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("my own skill")
    apply(plan([a for a in _agents(home) if a.key == "claude"], CMD, use_cli=False), home)
    assert "kernelport" in (mine / "SKILL.md").read_text()
    uninstall(home)
    assert (mine / "SKILL.md").read_text() == "my own skill"


def test_server_command_avoids_ephemeral_envs():
    assert server_command(Path("/home/u/.cache/uv/archive-v0/abc/bin/python"))[0] == "uvx"
    cmd = server_command(Path("/home/u/proj/.venv/bin/python"))
    assert cmd[1:] == ["-m", "kernelport", "mcp"]


# ------------------------------------------------------------------------- CLI

def test_cli_print_changes_nothing(home):
    before = _snapshot(home)
    res = CliRunner().invoke(app, ["setup", "--print", "--home", str(home)])
    assert res.exit_code == 0, res.output
    assert "Manual setup" in res.output and "mcpServers" in res.output and "mcp_servers.kernelport" in res.output
    assert _snapshot(home) == before


def test_cli_declined_changes_nothing(home):
    before = _snapshot(home)
    res = CliRunner().invoke(app, ["setup", "--home", str(home), "--no-agent-cli"], input="n\n")
    assert res.exit_code == 1 and "Planned changes" in res.output
    assert _snapshot(home) == before


def test_cli_install_and_uninstall(home):
    r = CliRunner()
    res = r.invoke(app, ["setup", "--home", str(home), "--no-agent-cli", "-a", "claude"], input="y\n")
    assert res.exit_code == 0, res.output
    assert "Restart your agent" in res.output and "port the CUDA kernels in this repo to Triton" in res.output
    assert "kernelport" in json.loads((home / ".claude.json").read_text())["mcpServers"]
    res = r.invoke(app, ["setup", "--uninstall", "--home", str(home)], input="y\n")
    assert res.exit_code == 0, res.output
    assert not (home / ".claude.json").exists()
    assert not (home / ".claude" / "skills" / "kernelport").exists()
