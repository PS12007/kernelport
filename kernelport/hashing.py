"""Content hashes for kernels, references, specs and ports.

Hashes ignore line-ending style: CRLF is folded to LF first. A ``.kernelport/``
workspace frozen on Linux or Colab and checked out on Windows (git
``core.autocrlf=true``) must still match its manifest, and a port verified on a
Mac must keep its PASS when the repo is pulled elsewhere. Any other byte change
still changes the hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """sha256 of ``data`` with CRLF folded to LF."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_text(text: str) -> str:
    """sha256 of UTF-8 text (line-ending insensitive)."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """sha256 of a file's contents (line-ending insensitive)."""
    return sha256_bytes(Path(path).read_bytes())
