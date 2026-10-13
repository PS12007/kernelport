"""result.json schema: the record of one verification run.

Status values:
- PASS        every case ran on real hardware and matched the frozen reference
- FAIL        at least one case mismatched or crashed
- UNVERIFIED  the port exists but could not be run here (backend unavailable)
- SKIPPED     not attempted (e.g. no port written)

Only PASS means "proven". Nothing else may be displayed as PASS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUSES = ("PASS", "FAIL", "UNVERIFIED", "SKIPPED")


@dataclass
class CaseResult:
    """Outcome of one (case, dtype)."""

    case: str
    dtype: str
    passed: bool
    input_hash: str
    outputs: list[dict[str, Any]] = field(default_factory=list)  # TensorCompare dicts
    error: str = ""  # crash / compile error text, truncated


@dataclass
class RunResult:
    """Outcome of verifying one kernel on one backend on one machine."""

    kernel: str
    backend: str
    status: str
    hardware: str
    reason: str = ""
    code_hash: str = ""
    reference_hash: str = ""
    spec_hash: str = ""
    tolerances: dict[str, dict] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    @property
    def num_passed(self) -> int:
        return sum(c.passed for c in self.cases)

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write as JSON, creating parent dirs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Inverse of to_dict, with validation."""
        d = dict(d)
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema {d.get('schema_version')}")
        d["cases"] = [CaseResult(**c) for c in d.get("cases", [])]
        return cls(**d)

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        """Read a result.json file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def failure_report(self, max_cases: int = 3) -> str:
        """Compact text for an agent: what failed, where, and by how much."""
        if self.status != "FAIL":
            return f"{self.kernel}/{self.backend}: {self.status} {self.reason}".strip()
        failed = [c for c in self.cases if not c.passed]
        lines = [f"{self.kernel}/{self.backend} on {self.hardware}: FAIL "
                 f"({self.num_passed}/{len(self.cases)} cases passed)"]
        if self.reason:
            lines.append(self.reason)
        for c in failed[:max_cases]:
            lines.append(f"- case {c.case} dtype {c.dtype}:")
            if c.error:
                lines.append(f"  error: {c.error}")
            for o in c.outputs:
                if o["passed"]:
                    continue
                lines.append(f"  {o['name']} shape {o['shape']}: {o['reason']}")
                if o["expected_slice"]:
                    lines.append(f"  flat[{o['slice_start']}:] expected {o['expected_slice']}")
                    lines.append(f"  flat[{o['slice_start']}:] actual   {o['actual_slice']}")
        if len(failed) > max_cases:
            lines.append(f"... and {len(failed) - max_cases} more failing cases: "
                         + ", ".join(f"{c.case}/{c.dtype}" for c in failed[max_cases:]))
        return "\n".join(lines)
