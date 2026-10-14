"""The ``.kernelport/`` workspace inside the user's repo.

Layout::

    .kernelport/
      index.json                       kernel id -> slug
      kernels/<slug>/
        kernel.json                    analyzer snapshot (id, file, line, params, features)
        original.cu                    only for kernels embedded in .py files
        launch.json                    candidate input spec
        reference.py                   candidate reference
        frozen/                        written ONLY by a passing validate_reference
          launch.json  reference.py  manifest.json
        validation.json                last validate_reference result
        ports/<backend>/
          port<ext>
          attempts.jsonl               one line per verify_port call
          results/<hardware>.json      latest result on each machine

Every path is built here from validated names and checked to stay inside
``.kernelport/``; nothing outside it is ever written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernelport.analyzer.models import KernelInfo
from kernelport.backends.base import BACKEND_NAMES, get_backend
from kernelport.harness.results import RunResult
from kernelport.harness.spec import LaunchSpec
from kernelport.hashing import sha256_file, sha256_text  # noqa: F401  (re-exported for tools)

WORKSPACE_DIR = ".kernelport"
MAX_CODE_BYTES = 512_000
MAX_ATTEMPTS = 6


class WorkspaceError(Exception):
    """A request the workspace refuses (bad id, unsafe path, rule violation)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe name: lowercase letters, digits, - and _."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()
    return (s or "x")[:max_len]


def find_root(start: Path | None = None) -> Path:
    """Repo root: $KERNELPORT_ROOT, else nearest ancestor with .kernelport or .git, else cwd."""
    env = os.environ.get("KERNELPORT_ROOT")
    if env:
        return Path(env).resolve()
    start = (start or Path.cwd()).resolve()
    for d in [start, *start.parents]:
        if (d / WORKSPACE_DIR).is_dir() or (d / ".git").exists():
            return d
    return start


@dataclass
class KernelRecord:
    """A kernel registered in the workspace."""

    id: str
    slug: str
    dir: Path
    info: dict[str, Any]

    @property
    def name(self) -> str:
        return self.info["name"]


class Workspace:
    """Read and write the .kernelport/ directory of one repo."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root).resolve() if root else find_root()
        self.dir = self.root / WORKSPACE_DIR

    # ---------------------------------------------------------- path safety
    def safe_path(self, *parts: str) -> Path:
        """Path inside .kernelport/ built from ``parts``; raises if it would escape."""
        for part in parts:
            if not part or part in (".", "..") or "/" in part or "\\" in part or ":" in part:
                raise WorkspaceError(f"unsafe path component {part!r}")
        p = self.dir.joinpath(*parts)
        resolved = p.resolve()
        base = self.dir.resolve()
        if resolved != base and base not in resolved.parents:
            raise WorkspaceError(f"path escapes the workspace: {p}")
        return p

    def _write(self, text: str, *parts: str) -> Path:
        p = self.safe_path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            raise WorkspaceError(f"refusing to write through symlink {p}")
        p.write_bytes(text.encode("utf-8"))  # no newline translation: hashes match file bytes
        return p

    def ensure(self) -> None:
        """Create .kernelport/ with a .gitignore-friendly README."""
        if not self.dir.exists():
            self.dir.mkdir(parents=True)
            (self.dir / "README.md").write_bytes(
                b"kernelport workspace: frozen references, input specs, ports, results.\n"
                b"Commit it to share verified ports; delete it to start over.\n")

    # ---------------------------------------------------------------- index
    def _index(self) -> dict[str, str]:
        p = self.dir / "index.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def _save_index(self, idx: dict[str, str]) -> None:
        self._write(json.dumps(idx, indent=2, sort_keys=True), "index.json")

    def register(self, info: KernelInfo) -> KernelRecord:
        """Add (or refresh) a scanned kernel. Returns its record."""
        self.ensure()
        idx = self._index()
        slug = idx.get(info.id)
        if slug is None:
            slug = f"{slugify(info.name, 32)}-{hashlib.sha1(info.id.encode()).hexdigest()[:6]}"
            idx[info.id] = slug
            self._save_index(idx)
        data = info.to_dict(include_source=False)
        data["source_hash"] = sha256_text(info.source)
        self._write(json.dumps(data, indent=2), "kernels", slug, "kernel.json")
        if info.file.endswith(".py"):
            self._write(info.source, "kernels", slug, "original.cu")
        return self.get(info.id)

    def kernel_ids(self) -> list[str]:
        """All registered kernel ids."""
        return sorted(self._index())

    def get(self, kernel_id: str) -> KernelRecord:
        """Record by full id, or by bare kernel name if that is unique."""
        idx = self._index()
        if kernel_id not in idx:
            matches = [k for k in idx if k.split("::")[-1].split("@")[0] == kernel_id]
            if len(matches) == 1:
                kernel_id = matches[0]
            elif matches:
                raise WorkspaceError(f"{kernel_id!r} is ambiguous: {matches}")
            else:
                raise WorkspaceError(f"unknown kernel {kernel_id!r}; run scan_kernels first")
        slug = idx[kernel_id]
        if not re.fullmatch(r"[a-z0-9_-]+", slug):
            raise WorkspaceError(f"corrupt index entry for {kernel_id!r}")
        kdir = self.safe_path("kernels", slug)
        info = json.loads((kdir / "kernel.json").read_text(encoding="utf-8"))
        return KernelRecord(kernel_id, slug, kdir, info)

    def original_source(self, rec: KernelRecord) -> Path:
        """File to compile for the original kernel."""
        if rec.info["file"].endswith(".py"):
            return rec.dir / "original.cu"
        path = (self.root / rec.info["file"]).resolve()
        if self.root not in path.parents:
            raise WorkspaceError(f"kernel source outside the repo: {path}")
        return path

    # -------------------------------------------------- spec and reference
    def write_spec(self, kernel_id: str, spec: dict) -> Path:
        """Validate and save the candidate launch.json."""
        from kernelport.analyzer.suggest import has_todo

        rec = self.get(kernel_id)
        if has_todo(spec):
            raise WorkspaceError("spec still contains TODO values; fill them in first")
        try:
            LaunchSpec.from_dict(spec)
        except (TypeError, ValueError, KeyError) as exc:
            raise WorkspaceError(f"invalid spec: {exc}") from exc
        return self._write(json.dumps(spec, indent=2), "kernels", rec.slug, "launch.json")

    def write_reference(self, kernel_id: str, code: str) -> Path:
        """Save the candidate reference. Does NOT touch the frozen copy."""
        rec = self.get(kernel_id)
        _check_code(code)
        try:
            compile(code, "reference.py", "exec")
        except SyntaxError as exc:
            raise WorkspaceError(f"reference has a syntax error: {exc}") from exc
        if not re.search(r"^def reference\s*\(", code, flags=re.M):
            raise WorkspaceError("reference code must define a top-level `def reference(...)`")
        return self._write(code, "kernels", rec.slug, "reference.py")

    def candidate(self, kernel_id: str) -> tuple[LaunchSpec, Path]:
        """(spec, reference path) of the candidate, for validation."""
        rec = self.get(kernel_id)
        spec_p, ref_p = rec.dir / "launch.json", rec.dir / "reference.py"
        if not spec_p.is_file():
            raise WorkspaceError("no input spec saved yet (write_reference with a spec)")
        if not ref_p.is_file():
            raise WorkspaceError("no reference saved yet (write_reference)")
        return LaunchSpec.load(spec_p), ref_p

    def freeze(self, kernel_id: str, validation: RunResult) -> dict:
        """Freeze the candidate after a PASSING validation. The only way to (re)freeze."""
        if validation.status != "PASS" or validation.backend != "cuda":
            raise WorkspaceError("only a passing validation against the original CUDA kernel can freeze")
        rec = self.get(kernel_id)
        spec_text = (rec.dir / "launch.json").read_bytes().decode("utf-8")
        ref_text = (rec.dir / "reference.py").read_bytes().decode("utf-8")
        if sha256_text(ref_text) != validation.reference_hash:
            raise WorkspaceError("reference changed during validation; validate again")
        manifest = {
            "kernel_id": rec.id,
            "reference_hash": sha256_text(ref_text),
            "spec_hash": sha256_text(spec_text),
            "source_hash": validation.code_hash,
            "validated_on": validation.hardware,
            "validated_at": validation.timestamp,
            "cases_passed": validation.num_passed,
        }
        self._write(spec_text, "kernels", rec.slug, "frozen", "launch.json")
        self._write(ref_text, "kernels", rec.slug, "frozen", "reference.py")
        self._write(json.dumps(manifest, indent=2), "kernels", rec.slug, "frozen", "manifest.json")
        return manifest

    def frozen(self, kernel_id: str) -> tuple[LaunchSpec, Path, dict]:
        """(spec, reference path, manifest) of the frozen reference, after an integrity check."""
        rec = self.get(kernel_id)
        fdir = rec.dir / "frozen"
        mpath = fdir / "manifest.json"
        if not mpath.is_file():
            raise WorkspaceError(f"reference for {rec.id} is not frozen; run validate_reference first")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if sha256_file(fdir / "reference.py") != manifest["reference_hash"] or \
                sha256_file(fdir / "launch.json") != manifest["spec_hash"]:
            raise WorkspaceError("frozen reference was modified after validation; run validate_reference again")
        return LaunchSpec.load(fdir / "launch.json"), fdir / "reference.py", manifest

    def is_frozen(self, kernel_id: str) -> bool:
        """True if a valid frozen reference exists."""
        try:
            self.frozen(kernel_id)
            return True
        except WorkspaceError:
            return False

    def pending_validation(self) -> list[str]:
        """Kernel ids with a candidate reference + spec that differs from the frozen copy (or none)."""
        out = []
        for kid in self.kernel_ids():
            d = self.get(kid).dir
            spec_p, ref_p = d / "launch.json", d / "reference.py"
            if not (spec_p.is_file() and ref_p.is_file()):
                continue
            try:
                _, _, manifest = self.frozen(kid)
            except WorkspaceError:
                out.append(kid)
                continue
            if (sha256_file(ref_p), sha256_file(spec_p)) != (manifest["reference_hash"], manifest["spec_hash"]):
                out.append(kid)
        return out

    # ---------------------------------------------------------------- ports
    def port_path(self, kernel_id: str, backend: str) -> Path:
        """Where the port for ``backend`` lives (may not exist yet)."""
        if backend not in BACKEND_NAMES:
            raise WorkspaceError(f"unknown backend {backend!r}; known: {', '.join(BACKEND_NAMES)}")
        rec = self.get(kernel_id)
        ext = get_backend(backend).config.extension
        return self.safe_path("kernels", rec.slug, "ports", backend, f"port{ext}")

    def write_port(self, kernel_id: str, backend: str, code: str) -> Path:
        """Save a port. Any earlier PASS for the old code stops applying (hash mismatch)."""
        path = self.port_path(kernel_id, backend)
        _check_code(code)
        if path.suffix == ".py":
            try:
                compile(code, path.name, "exec")
            except SyntaxError as exc:
                raise WorkspaceError(f"port has a syntax error: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._write(code, *path.relative_to(self.dir).parts)

    def log_attempt(self, kernel_id: str, backend: str, result: RunResult) -> dict:
        """Append one verify_port call to attempts.jsonl and return the entry."""
        path = self.port_path(kernel_id, backend).parent / "attempts.jsonl"
        attempts = self.attempts(kernel_id, backend)
        entry = {
            "attempt": len(attempts) + 1,
            "timestamp": _now(),
            "code_hash": result.code_hash,
            "reference_hash": result.reference_hash,
            "status": result.status,
            "hardware": result.hardware,
            "cases_passed": result.num_passed,
            "cases_total": len(result.cases),
            "reason": result.reason,
        }
        self.safe_path(*path.relative_to(self.dir).parts)
        with path.open("a", encoding="utf-8", newline="\n") as f:  # LF on every OS
            f.write(json.dumps(entry) + "\n")
        return entry

    def attempts(self, kernel_id: str, backend: str) -> list[dict]:
        """All logged attempts for a port."""
        path = self.port_path(kernel_id, backend).parent / "attempts.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def failures_since_pass(self, kernel_id: str, backend: str) -> int:
        """FAIL attempts after the most recent PASS (the repair budget used so far)."""
        n = 0
        for a in self.attempts(kernel_id, backend):
            n = 0 if a["status"] == "PASS" else n + (a["status"] == "FAIL")
        return n

    def save_result(self, kernel_id: str, backend: str, result: RunResult) -> Path:
        """Store the latest result for this backend on this hardware."""
        port_dir = self.port_path(kernel_id, backend).parent
        path = port_dir / "results" / f"{slugify(result.hardware, 64)}.json"
        self.safe_path(*path.relative_to(self.dir).parts)
        result.save(path)
        return path

    def results(self, kernel_id: str, backend: str) -> list[RunResult]:
        """Latest result per hardware for a port."""
        rdir = self.port_path(kernel_id, backend).parent / "results"
        if not rdir.is_dir():
            return []
        return [RunResult.load(p) for p in sorted(rdir.glob("*.json"))]

    def save_validation(self, kernel_id: str, result: RunResult) -> Path:
        """Store the last validate_reference result."""
        rec = self.get(kernel_id)
        p = self.safe_path("kernels", rec.slug, "validation.json")
        result.save(p)
        return p

    # --------------------------------------------------------------- status
    def status(self) -> list[dict]:
        """One row per kernel x backend x hardware, with honest, current statuses.

        A stored PASS only counts if the port and frozen reference still have the
        hashes it was earned with; otherwise the row is UNVERIFIED with a reason.
        """
        rows = []
        for kid in self.kernel_ids():
            rec = self.get(kid)
            frozen_hash = None
            try:
                _, _, manifest = self.frozen(kid)
                frozen_hash = manifest["reference_hash"]
                ref_state = "frozen"
            except WorkspaceError as exc:
                ref_state = "not frozen" if "not frozen" in str(exc) else "tampered"
            pdir = rec.dir / "ports"
            backends = sorted(p.name for p in pdir.iterdir() if p.is_dir()) if pdir.is_dir() else []
            if not backends:
                rows.append({"kernel": kid, "reference": ref_state, "backend": "-", "hardware": "-",
                             "status": "SKIPPED", "reason": "no port written", "cases": ""})
            for b in backends:
                port = self.port_path(kid, b)
                code_hash = sha256_file(port) if port.is_file() else None
                results = self.results(kid, b)
                if not results:
                    rows.append({"kernel": kid, "reference": ref_state, "backend": b, "hardware": "-",
                                 "status": "UNVERIFIED", "reason": "not verified on any hardware yet",
                                 "cases": ""})
                for r in results:
                    status, reason = r.status, r.reason
                    if code_hash is None:
                        status, reason = "SKIPPED", "port file missing"
                    elif r.code_hash != code_hash:
                        status, reason = "UNVERIFIED", "port changed since this result"
                    elif r.status in ("PASS", "FAIL") and r.reference_hash != frozen_hash:
                        status, reason = "UNVERIFIED", "reference changed or not frozen since this result"
                    rows.append({"kernel": kid, "reference": ref_state, "backend": b,
                                 "hardware": r.hardware, "status": status, "reason": reason,
                                 "cases": f"{r.num_passed}/{len(r.cases)}" if r.cases else "",
                                 "timestamp": r.timestamp})
        return rows


def _check_code(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise WorkspaceError("code is empty")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise WorkspaceError(f"code is larger than {MAX_CODE_BYTES} bytes")
