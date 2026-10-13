"""Plan, apply and undo kernelport's changes to agent configs.

Nothing is written until ``apply`` is called; the CLI shows ``plan`` first and asks
for approval. Every file is backed up before it is touched, and the install state
records what was added (and any value it replaced) so uninstall undoes exactly that.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kernelport.setup_agents.backup import backup, load_state, new_backup_dir, save_state
from kernelport.setup_agents.detect import SERVER_NAME, Agent
from kernelport.setup_agents.skills_install import install_skill, remove_skill, skill_files

TOML_BEGIN = "# >>> kernelport (added by `kernelport setup`; remove with `kernelport setup --uninstall`) >>>"
TOML_END = "# <<< kernelport <<<"


class SetupError(Exception):
    """A change kernelport refuses to make."""


@dataclass
class Change:
    """One planned change, shown to the user before anything is written."""

    agent: str
    kind: str  # "mcp" or "skill"
    path: Path
    action: str  # "add", "update", "unchanged", "remove"
    preview: str  # exact JSON/TOML snippet, command, or file list
    method: str = "file"  # "file" or "cli" (Claude Code's `claude mcp add`)
    extra: dict = field(default_factory=dict)


def server_entry(agent: Agent, command: list[str]) -> dict:
    """The MCP server entry for one agent, in that agent's JSON shape."""
    entry: dict = {"command": command[0], "args": command[1:]}
    if agent.key in ("claude", "cursor"):
        entry = {"type": "stdio", **entry}
    if agent.workspace_env:
        entry["env"] = dict(agent.workspace_env)
    return entry


def toml_block(command: list[str], env: dict[str, str]) -> str:
    """Marker-delimited Codex TOML block. json.dumps output is a valid TOML basic string."""
    lines = [TOML_BEGIN, f"[mcp_servers.{SERVER_NAME}]", f"command = {json.dumps(command[0])}",
             f"args = [{', '.join(json.dumps(a) for a in command[1:])}]"]
    if env:
        lines += [f"[mcp_servers.{SERVER_NAME}.env]"] + [f"{k} = {json.dumps(v)}" for k, v in env.items()]
    lines.append(TOML_END)
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SetupError(f"{path} is not valid JSON ({exc}); fix it or use `kernelport setup --print`") from exc
    if not isinstance(data, dict):
        raise SetupError(f"{path} does not contain a JSON object")
    return data


def _strip_block(text: str) -> str:
    return re.sub(rf"\n?{re.escape(TOML_BEGIN)}.*?{re.escape(TOML_END)}\n?", "\n", text, flags=re.S)


# ---------------------------------------------------------------------- plan

def plan(agents: list[Agent], command: list[str], use_cli: bool = True) -> list[Change]:
    """Changes needed to register the MCP server and install the skill for each agent."""
    changes: list[Change] = []
    for a in agents:
        if a.config_format == "json":
            entry = server_entry(a, command)
            current = _read_json(a.config_path).get("mcpServers", {}).get(SERVER_NAME)
            action = "unchanged" if current == entry else ("update" if current else "add")
            cli = use_cli and a.key == "claude" and shutil.which(a.cli) is not None
            if cli:
                preview = " ".join(["claude", "mcp", "add", "--transport", "stdio", "--scope", "user",
                                    SERVER_NAME, "--", *(_quote(c) for c in command)])
            else:
                preview = json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2)
            changes.append(Change(a.key, "mcp", a.config_path, action, preview,
                                  "cli" if cli else "file", {"entry": entry, "previous": current}))
        else:
            text = a.config_path.read_text(encoding="utf-8") if a.config_path.is_file() else ""
            block = toml_block(command, a.workspace_env)
            if TOML_BEGIN in text:
                action = "unchanged" if block.strip() in text else "update"
            else:
                try:
                    parsed = tomllib.loads(text) if text else {}
                except tomllib.TOMLDecodeError as exc:
                    raise SetupError(f"{a.config_path} is not valid TOML ({exc})") from exc
                if SERVER_NAME in parsed.get("mcp_servers", {}):
                    raise SetupError(f"{a.config_path} already has an [mcp_servers.{SERVER_NAME}] table not "
                                     "added by kernelport; remove it or use `kernelport setup --print`")
                action = "add"
            changes.append(Change(a.key, "mcp", a.config_path, action, block, "file", {"block": block}))
        files = "\n".join(f"  {a.skill_dir / f}" for f in skill_files())
        changes.append(Change(a.key, "skill", a.skill_dir, "update" if a.skill_dir.exists() else "add",
                              f"copy kernelport skill ({len(skill_files())} files):\n{files}"))
    return changes


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s else s


# --------------------------------------------------------------------- apply

def apply(changes: list[Change], home: Path) -> Path:
    """Back up every touched path, then make the changes. Returns the backup dir."""
    todo = [c for c in changes if c.action != "unchanged"]
    dest = new_backup_dir(home)
    manifest = backup(sorted({c.path for c in todo}, key=str), dest)
    backups = {f["original"]: f["backup"] for f in manifest["files"]}
    state = load_state(home)
    for c in todo:
        st = state.setdefault(c.agent, {})
        if c.kind == "mcp" and "original_backup" not in st:
            st["original_backup"] = backups.get(str(c.path))  # None if the file did not exist
        if c.kind == "skill":
            if c.path.exists() and not st.get("skill_installed"):
                st["skill_previous_backup"] = str(dest)  # a non-kernelport dir was replaced
            install_skill(c.path)
            st.update({"skill_installed": True, "skill_dir": str(c.path)})
        elif c.path.suffix == ".toml":
            text = c.path.read_text(encoding="utf-8") if c.path.is_file() else ""
            new = _strip_block(text).rstrip("\n")
            new = (new + "\n\n" if new else "") + c.extra["block"]
            tomllib.loads(new)  # never write invalid TOML
            c.path.parent.mkdir(parents=True, exist_ok=True)
            _write(c.path, new)
            st.update({"mcp_installed": True, "config_path": str(c.path), "format": "toml",
                       "created_file": not bool(text)})
        else:
            if "mcp_installed" not in st:
                st["previous_entry"] = c.extra["previous"]
                st["created_file"] = not c.path.is_file()
            if c.method == "cli":
                if c.extra["previous"] is not None:
                    _run(["claude", "mcp", "remove", SERVER_NAME, "--scope", "user"])
                e = c.extra["entry"]
                _run(["claude", "mcp", "add", "--transport", "stdio", "--scope", "user", SERVER_NAME, "--",
                      e["command"], *e["args"]])
            else:
                data = _read_json(c.path)
                data.setdefault("mcpServers", {})[SERVER_NAME] = c.extra["entry"]
                c.path.parent.mkdir(parents=True, exist_ok=True)
                _write(c.path, json.dumps(data, indent=2) + "\n")
            st.update({"mcp_installed": True, "config_path": str(c.path), "format": "json", "method": c.method})
        st["backup_dir"] = str(dest)
    save_state(home, state)
    return dest


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))  # LF endings on every OS


def _same_as_original(st: dict, text: str | None = None, data: dict | None = None) -> bytes | None:
    """Original bytes if the cleaned-up config equals the pre-install file, else None."""
    orig = st.get("original_backup")
    if not orig or not Path(orig).is_file():
        return None
    raw = Path(orig).read_bytes()
    if data is not None:
        try:
            return raw if json.loads(raw.decode("utf-8")) == data else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    def norm(t: str) -> str:
        return t.replace("\r\n", "\n").strip("\n")

    return raw if norm(raw.decode("utf-8")) == norm(text or "") else None


def _run(cmd: list[str]) -> None:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise SetupError(f"{cmd[0]} not found on PATH")
    res = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True)
    if res.returncode != 0:
        raise SetupError(f"`{' '.join(cmd)}` failed: {res.stderr.strip() or res.stdout.strip()}")


# ----------------------------------------------------------------- uninstall

def plan_uninstall(home: Path) -> list[Change]:
    """What --uninstall would undo, from the recorded install state."""
    changes = []
    for agent, st in load_state(home).items():
        if st.get("mcp_installed"):
            prev = st.get("previous_entry")
            how = "restore the previous kernelport entry" if prev else f"remove the {SERVER_NAME} server entry"
            if st.get("format") == "toml":
                how = "remove the marked kernelport block"
            changes.append(Change(agent, "mcp", Path(st["config_path"]), "remove", how, st.get("method", "file")))
        if st.get("skill_installed"):
            how = "delete the installed skill" + (" and restore the directory it replaced"
                                                 if st.get("skill_previous_backup") else "")
            changes.append(Change(agent, "skill", Path(st["skill_dir"]), "remove", how))
    return changes


def uninstall(home: Path) -> Path | None:
    """Undo everything recorded in the install state. Backs up current files first."""
    state = load_state(home)
    changes = plan_uninstall(home)
    if not changes:
        return None
    dest = new_backup_dir(home)
    backup(sorted({c.path for c in changes}, key=str), dest)
    for agent, st in list(state.items()):
        if st.get("mcp_installed"):
            path = Path(st["config_path"])
            if st.get("format") == "toml":
                if path.is_file():
                    text = _strip_block(path.read_text(encoding="utf-8")).strip("\n")
                    original = _same_as_original(st, text=text)
                    if not text and st.get("created_file"):
                        path.unlink()
                    elif original is not None:
                        path.write_bytes(original)  # byte-exact restore when nothing else changed
                    else:
                        tomllib.loads(text)
                        _write(path, text + "\n")
            elif st.get("method") == "cli" and st.get("previous_entry") is None:
                _run(["claude", "mcp", "remove", SERVER_NAME, "--scope", "user"])
            elif path.is_file():
                data = _read_json(path)
                servers = data.get("mcpServers", {})
                if st.get("previous_entry") is not None:
                    servers[SERVER_NAME] = st["previous_entry"]
                else:
                    servers.pop(SERVER_NAME, None)
                    if not servers:
                        data.pop("mcpServers", None)
                original = _same_as_original(st, data=data)
                if not data and st.get("created_file"):
                    path.unlink()
                elif original is not None:
                    path.write_bytes(original)  # byte-exact restore when nothing else changed
                else:
                    _write(path, json.dumps(data, indent=2) + "\n")
        if st.get("skill_installed"):
            skill = Path(st["skill_dir"])
            remove_skill(skill)
            prev = st.get("skill_previous_backup")
            if prev:
                _restore_from_backup(Path(prev), skill)
        del state[agent]
    save_state(home, state)
    return dest


def _restore_from_backup(backup_dir: Path, original: Path) -> None:
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    for f in manifest["files"]:
        if Path(f["original"]) == original and f["backup"]:
            shutil.copytree(f["backup"], original)
