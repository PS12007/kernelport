"""Back up files before kernelport changes them, and remember what was changed.

Backups live in ~/.kernelport/backups/<timestamp>/ with a manifest. The install
state (what kernelport added to each agent, and any value it replaced) lives in
~/.kernelport/setup_state.json so ``kernelport setup --uninstall`` can undo
exactly that and nothing else.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


def kp_home(home: Path) -> Path:
    """~/.kernelport (kernelport's own per-user dir, not the per-repo workspace)."""
    return Path(home) / ".kernelport"


def new_backup_dir(home: Path) -> Path:
    """Create a fresh, timestamped backup directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d = kp_home(home) / "backups" / stamp
    d.mkdir(parents=True, exist_ok=False)
    return d


def backup(paths: list[Path], dest: Path) -> dict:
    """Copy each existing file/dir into ``dest``; write and return the manifest."""
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "files": []}
    for i, p in enumerate(paths):
        p = Path(p)
        entry = {"original": str(p), "existed": p.exists(), "backup": None}
        if p.is_file():
            target = dest / f"{i:02d}_{p.name}"
            shutil.copy2(p, target)
            entry["backup"] = str(target)
        elif p.is_dir():
            target = dest / f"{i:02d}_{p.name}"
            shutil.copytree(p, target)
            entry["backup"] = str(target)
        manifest["files"].append(entry)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_state(home: Path) -> dict:
    """Current install state ({} if none)."""
    p = kp_home(home) / "setup_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(home: Path, state: dict) -> None:
    """Persist install state."""
    p = kp_home(home) / "setup_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
