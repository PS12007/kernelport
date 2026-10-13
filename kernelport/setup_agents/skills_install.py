"""Copy the kernelport skill into an agent's skills directory (and remove it again)."""

from __future__ import annotations

import shutil
from pathlib import Path

from kernelport.tools import skills_dir


def skill_files() -> list[Path]:
    """Files that make up the skill, relative to the skill root."""
    root = skills_dir()
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def install_skill(target: Path) -> list[str]:
    """Copy the skill tree to ``target`` (replacing it). Returns the files written."""
    src = skills_dir()
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return [str(target / f) for f in skill_files()]


def remove_skill(target: Path) -> bool:
    """Delete an installed skill directory. Returns True if something was removed."""
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False
