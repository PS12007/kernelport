"""Known coding agents: where their MCP config and skills live.

Locations follow each agent's official docs (checked 2026-10-13):
- Claude Code: MCP in ~/.claude.json "mcpServers" (user scope; `claude mcp add --scope user`),
  skills in ~/.claude/skills/<name>/SKILL.md
- Codex: MCP in ~/.codex/config.toml [mcp_servers.<name>], skills in ~/.agents/skills/
- Cursor: MCP in ~/.cursor/mcp.json "mcpServers" (supports ${workspaceFolder}), skills in ~/.cursor/skills/
- Gemini CLI: MCP in ~/.gemini/settings.json "mcpServers", skills in ~/.gemini/skills/
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SERVER_NAME = "kernelport"
SKILL_NAME = "kernelport"


@dataclass(frozen=True)
class Agent:
    """One supported agent on this machine."""

    key: str
    label: str
    config_path: Path
    config_format: str  # "json" or "toml"
    skill_dir: Path  # where skills/kernelport/ is copied
    cli: str  # executable name
    detected: bool
    workspace_env: dict[str, str]  # extra env so the server finds the open project

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {"key": self.key, "label": self.label, "config_path": str(self.config_path),
                "config_format": self.config_format, "skill_dir": str(self.skill_dir),
                "detected": self.detected}


def _on_path(name: str) -> bool:
    return shutil.which(name) is not None


def agents(home: Path | None = None, check_path: bool = True) -> list[Agent]:
    """All supported agents, with ``detected`` set for the ones present here."""
    home = Path(home) if home else Path.home()
    which = _on_path if check_path else (lambda _: False)
    return [
        Agent("claude", "Claude Code", home / ".claude.json", "json",
              home / ".claude" / "skills" / SKILL_NAME, "claude",
              which("claude") or (home / ".claude").is_dir() or (home / ".claude.json").is_file(), {}),
        Agent("codex", "Codex CLI", home / ".codex" / "config.toml", "toml",
              home / ".agents" / "skills" / SKILL_NAME, "codex",
              which("codex") or (home / ".codex").is_dir(), {}),
        Agent("cursor", "Cursor", home / ".cursor" / "mcp.json", "json",
              home / ".cursor" / "skills" / SKILL_NAME, "cursor",
              which("cursor") or (home / ".cursor").is_dir(),
              {"KERNELPORT_ROOT": "${workspaceFolder}"}),
        Agent("gemini", "Gemini CLI", home / ".gemini" / "settings.json", "json",
              home / ".gemini" / "skills" / SKILL_NAME, "gemini",
              which("gemini") or (home / ".gemini").is_dir(), {}),
    ]


def _ephemeral(exe: Path) -> bool:
    """True when running from a throwaway env (uvx / pipx run), which may be deleted later."""
    s = str(exe).replace("\\", "/").lower()
    return any(m in s for m in ("/uv/cache/", "/archive-v0/", "/.cache/pipx/", "/pipx/.cache/", "/uv/archive"))


def server_command(exe: Path | None = None) -> list[str]:
    """Command agents should run to start the kernelport MCP server."""
    exe = Path(exe or sys.executable)
    if _ephemeral(exe):
        return ["uvx", "kernelport", "mcp"]
    return [str(exe), "-m", "kernelport", "mcp"]
