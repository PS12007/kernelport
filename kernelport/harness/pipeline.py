"""Pipelines: several kernels run in a chain, each fed the previous kernel's real output.

Single-kernel verification tests each port on generated inputs. In a real program a kernel
reads what the kernel before it wrote: softmax outputs near 1/V, a layernorm's unit-variance
rows, a gradient buffer that already holds values. A pipeline checks the ports on that data.

A pipeline is plain JSON (``.kernelport/pipelines/<name>.json``)::

    {
      "name": "gpt2_loss_head",
      "buffers": {
        "logits":  {"dtype": "float32", "shape": ["B", "T", "V"], "init": {"dist": "normal", "std": 2.0}},
        "targets": {"dtype": "int32",   "shape": ["B", "T"], "init": {"dist": "randint", "high": "V"}},
        "probs":   {"dtype": "float32", "shape": ["B", "T", "V"]},
        "losses":  {"dtype": "float32", "shape": ["B", "T"]}
      },
      "stages": [
        {"kernel": "softmax_forward_online_kernel8", "dims": {"N": "B * T", "C": "V"},
         "bind": {"inp": "logits", "out": "probs"}},
        {"kernel": "crossentropy_forward_kernel1"}
      ],
      "cases": [{"name": "small", "dims": {"B": 2, "T": 3, "V": 300}}]
    }

- ``buffers`` hold the data. A buffer with ``init`` is generated like a launch.json input; a
  buffer without one must be written by a stage before any stage reads it.
- ``stages`` run in order. ``kernel`` is a kernel id with a frozen reference. ``dims`` maps the
  kernel's launch.json dims to expressions over the pipeline's dims (others pass through by
  name). ``bind`` maps the kernel's tensor params to buffers (same name if omitted). A buffer
  is reshaped to the param's shape, so element counts must match; dtypes must match exactly.
  ``dtype`` picks the kernel's ``T`` when the bound buffers don't decide it.

Checks, for every case:

1. **Stage checks (decide PASS/FAIL).** Each stage's port output is compared with its frozen
   reference run on the same inputs the port received, with the usual per-dtype tolerance.
2. **End-to-end drift (reported, never decides).** The chain of references in float64 from the
   same starting buffers, compared with the chain of ports. Rounding adds up over a chain, so
   this can exceed one kernel's tolerance with every stage correct; it is shown so the size of
   the accumulated error is known. Tolerances stay kernelport's, never the pipeline author's.

Backend ``cuda`` runs the original kernels instead of ports: it checks that the originals (and
references) agree on chained data, which can break a precondition that the single-kernel cases
never hit.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from kernelport.harness.compare import compare_tensor
from kernelport.harness.expr import evaluate, evaluate_int
from kernelport.harness.inputs import (NUMERIC_INIT_KEYS, TORCH_DTYPES, CaseData, _as_scalar_dtype,
                                       _generate, _sentinel, tensor_bytes)
from kernelport.harness.reference import run_reference
from kernelport.harness.results import CaseNotRunnable
from kernelport.harness.spec import ALL_DTYPES, CaseSpec, LaunchSpec
from kernelport.harness.tolerances import Tolerance, tolerance_for

PIPELINE_SCHEMA_VERSION = 1
MAX_STAGES = 32
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class PipelineError(ValueError):
    """The pipeline file is inconsistent with itself or with the kernels it names."""


# ---------------------------------------------------------------------------------- spec
@dataclass
class BufferSpec:
    """One named array the stages read and write."""

    name: str
    dtype: str
    shape: list[Any]
    init: dict | None = None  # None: must be written by a stage before it is read

    def resolved_shape(self, dims: dict[str, Any]) -> tuple[int, ...]:
        return tuple(evaluate_int(s, dims) for s in self.shape)


@dataclass
class StageSpec:
    """One kernel call."""

    kernel: str
    dims: dict[str, Any] = field(default_factory=dict)
    bind: dict[str, str] = field(default_factory=dict)
    dtype: str | None = None


@dataclass
class PipelineSpec:
    """A chain of kernel calls over shared buffers, plus the cases (pipeline dims) to run."""

    name: str
    buffers: dict[str, BufferSpec]
    stages: list[StageSpec]
    cases: list[CaseSpec]
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown = set(d) - {"name", "buffers", "stages", "cases", "seed"}
        if unknown:
            raise PipelineError(f"unknown pipeline keys {sorted(unknown)}")
        try:
            bufs = {n: BufferSpec(name=n, **b) for n, b in d["buffers"].items()}
            stages = [StageSpec(**s) for s in d["stages"]]
            cases = [CaseSpec(**c) for c in d["cases"]]
            spec = cls(name=d["name"], buffers=bufs, stages=stages, cases=cases, seed=int(d.get("seed", 0)))
        except KeyError as exc:
            raise PipelineError(f"pipeline is missing {exc}") from None
        except TypeError as exc:
            raise PipelineError(f"bad pipeline entry: {exc}") from None
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: Path) -> "PipelineSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {"name": self.name,
                "buffers": {n: {k: v for k, v in asdict(b).items() if k != "name"} for n, b in self.buffers.items()},
                "stages": [asdict(s) for s in self.stages],
                "cases": [asdict(c) for c in self.cases], "seed": self.seed}

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def validate(self) -> None:
        """Checks that need no kernel specs (those are in :func:`resolve`)."""
        if not NAME_RE.fullmatch(self.name):
            raise PipelineError(f"pipeline name {self.name!r}: use letters, digits, - and _ (max 64)")
        if not self.stages:
            raise PipelineError("pipeline has no stages")
        if len(self.stages) > MAX_STAGES:
            raise PipelineError(f"pipeline has more than {MAX_STAGES} stages")
        if not self.cases:
            raise PipelineError("pipeline has no cases")
        if len({c.name for c in self.cases}) != len(self.cases):
            raise PipelineError("case names must be unique")
        for b in self.buffers.values():
            if b.dtype not in ALL_DTYPES:
                raise PipelineError(f"buffer {b.name}: unknown dtype {b.dtype!r}")
            for c in self.cases:
                try:
                    b.resolved_shape(c.dims)
                except Exception as exc:
                    raise PipelineError(f"buffer {b.name}, case {c.name}: shape {b.shape}: {exc}") from None
        for s in self.stages:
            if s.dtype is not None and s.dtype not in ALL_DTYPES:
                raise PipelineError(f"stage {s.kernel}: unknown dtype {s.dtype!r}")
            for p, buf in s.bind.items():
                if buf not in self.buffers:
                    raise PipelineError(f"stage {s.kernel}: param {p} bound to unknown buffer {buf!r}")


@dataclass
class ResolvedStage:
    """A stage with its kernel's frozen spec and reference, and the binding worked out."""

    index: int
    kernel_id: str
    spec: LaunchSpec
    reference: Path
    dtype: str
    bind: dict[str, str]  # every tensor param -> buffer
    dims_expr: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.index + 1}:{self.spec.name}"

    def dims(self, case_dims: dict[str, Any]) -> dict[str, Any]:
        """The kernel's dims for one pipeline case."""
        return {**case_dims, **{k: evaluate(v, case_dims) for k, v in self.dims_expr.items()}}


def resolve(pipe: PipelineSpec, load_kernel: Callable[[str], tuple[str, LaunchSpec, Path]]) -> list[ResolvedStage]:
    """Check every stage against its kernel's frozen spec. ``load_kernel(id) -> (full id, spec, reference)``."""
    out: list[ResolvedStage] = []
    written: set[str] = {n for n, b in pipe.buffers.items() if b.init is not None}
    for i, st in enumerate(pipe.stages):
        where = f"stage {i + 1} ({st.kernel})"
        kid, spec, ref = load_kernel(st.kernel)
        tensor_params = [p for p in spec.params if p.is_tensor]
        names = {p.name for p in tensor_params}
        extra = set(st.bind) - names
        if extra:
            raise PipelineError(f"{where}: bind names params the kernel does not have: {sorted(extra)}; "
                                f"its tensor params are {sorted(names)}")
        bind = {}
        for p in tensor_params:
            buf = st.bind.get(p.name, p.name)
            if buf not in pipe.buffers:
                raise PipelineError(f"{where}: param {p.name} is not bound; add \"bind\": {{\"{p.name}\": "
                                    f"\"<buffer>\"}} or a buffer named {p.name}")
            bind[p.name] = buf
        # T from the buffers bound to T params, else the stage's dtype, else the spec's only dtype.
        t_dtypes = {pipe.buffers[bind[p.name]].dtype for p in tensor_params if p.dtype == "T"}
        if len(t_dtypes) > 1:
            raise PipelineError(f"{where}: buffers bound to the kernel's T params have different dtypes {sorted(t_dtypes)}")
        dtype = st.dtype or (t_dtypes.pop() if t_dtypes else None) or (spec.dtypes[0] if len(spec.dtypes) == 1 else None)
        if dtype is None:
            raise PipelineError(f"{where}: cannot tell which of {spec.dtypes} to run; set \"dtype\"")
        if dtype not in spec.dtypes:
            raise PipelineError(f"{where}: dtype {dtype} is not in the kernel's verified dtypes {spec.dtypes}")
        for p in tensor_params:
            buf = pipe.buffers[bind[p.name]]
            want = p.resolved_dtype(dtype)
            if buf.dtype != want:
                raise PipelineError(f"{where}: param {p.name} is {want} but buffer {buf.name} is {buf.dtype}")
            if p.kind in ("input", "inout") and buf.name not in written:
                raise PipelineError(f"{where}: reads buffer {buf.name} before any stage writes it (give it an "
                                    f"\"init\" or move the stage that writes it earlier)")
        for p in tensor_params:
            if p.is_output:
                written.add(bind[p.name])
        rs = ResolvedStage(i, kid, spec, ref, dtype, bind, dict(st.dims))
        for c in pipe.cases:
            try:
                kd = rs.dims(c.dims)
                for p in spec.params:
                    if p.is_tensor:
                        n = _numel(p.resolved_shape(kd))
                        m = _numel(pipe.buffers[bind[p.name]].resolved_shape(c.dims))
                        if n != m:
                            raise PipelineError(f"param {p.name} has {n} elements, buffer {bind[p.name]} has {m}")
                    else:
                        p.resolved_value(kd)
            except PipelineError as exc:
                raise PipelineError(f"{where}, case {c.name}: {exc}") from None
            except Exception as exc:
                raise PipelineError(f"{where}, case {c.name}: {type(exc).__name__}: {exc} "
                                    f"(map the kernel's dims in the stage's \"dims\")") from None
        out.append(rs)
    return out


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


# ------------------------------------------------------------------------------- running
def initial_buffers(pipe: PipelineSpec, case: CaseSpec) -> tuple[dict[str, torch.Tensor], str]:
    """Starting buffers for a case (generated or sentinel-filled) and their hash."""
    bufs: dict[str, torch.Tensor] = {}
    h = hashlib.sha256()
    for name in sorted(pipe.buffers):
        b = pipe.buffers[name]
        shape = b.resolved_shape(case.dims)
        if b.init is None:
            t = _sentinel(shape, b.dtype)
        else:
            init = {**b.init, **case.init.get(name, {})}
            init = {k: (evaluate(v, case.dims) if k in NUMERIC_INIT_KEYS and isinstance(v, str) else v)
                    for k, v in init.items()}
            seed = (pipe.seed * 1_000_003 + zlib.crc32(f"{pipe.name}|{case.name}|{name}".encode())) & 0xFFFFFFFF
            t = _generate(init, shape, b.dtype, seed)
        bufs[name] = t
        h.update(f"{name}:{b.dtype}:{shape}".encode())
        h.update(tensor_bytes(t))
    return bufs, h.hexdigest()


def stage_data(rs: ResolvedStage, bufs: dict[str, torch.Tensor], case_name: str,
               case_dims: dict[str, Any]) -> CaseData:
    """CaseData for one stage, viewing the pipeline's buffers in the kernel's shapes.

    Outputs start as sentinels (NaN / INT_SENTINEL) so unwritten elements show up, exactly as in
    single-kernel verification. Inputs and inouts are copies, so a stage cannot change a buffer
    except through its returned outputs.
    """
    dims = rs.dims(case_dims)
    tensors, scalars = {}, {}
    for p in rs.spec.params:
        if not p.is_tensor:
            scalars[p.name] = _as_scalar_dtype(p.resolved_value(dims), p.dtype)
            continue
        shape = p.resolved_shape(dims)
        if p.kind == "output":
            tensors[p.name] = _sentinel(shape, p.resolved_dtype(rs.dtype))
        else:
            tensors[p.name] = bufs[rs.bind[p.name]].reshape(shape).clone()
    return CaseData(case_name, rs.dtype, dims, tensors, scalars, "")


def _store(rs: ResolvedStage, bufs: dict[str, torch.Tensor], outs: dict[str, torch.Tensor]) -> None:
    for p in rs.spec.params:
        if p.is_output and p.name in outs:
            buf = bufs[rs.bind[p.name]]
            t = outs[p.name].detach().cpu()
            if t.numel() == buf.numel():
                bufs[rs.bind[p.name]] = t.to(buf.dtype).reshape(buf.shape)


@dataclass
class StageResult:
    stage: str  # "2:crossentropy_forward_kernel1"
    kernel_id: str
    passed: bool
    outputs: list[dict] = field(default_factory=list)  # TensorCompare dicts
    error: str = ""


@dataclass
class PipelineCaseResult:
    case: str
    passed: bool
    input_hash: str
    stages: list[StageResult] = field(default_factory=list)
    end_to_end: list[dict] = field(default_factory=list)  # TensorCompare dicts, informational
    not_runnable: str = ""


@dataclass
class PipelineResult:
    pipeline: str
    backend: str
    status: str  # PASS FAIL UNVERIFIED SKIPPED
    hardware: str
    reason: str = ""
    pipeline_hash: str = ""
    code_hashes: dict[str, str] = field(default_factory=dict)  # kernel id -> port (or original) hash
    reference_hashes: dict[str, str] = field(default_factory=dict)
    cases: list[PipelineCaseResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema_version: int = PIPELINE_SCHEMA_VERSION

    @property
    def num_passed(self) -> int:
        return sum(c.passed for c in self.cases)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(self.to_dict(), indent=2).encode("utf-8"))

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineResult":
        d = dict(d)
        if d.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise ValueError(f"unsupported pipeline result schema {d.get('schema_version')}")
        cases = []
        for c in d.get("cases", []):
            c = dict(c)
            c["stages"] = [StageResult(**s) for s in c.get("stages", [])]
            cases.append(PipelineCaseResult(**c))
        d["cases"] = cases
        return cls(**d)

    @classmethod
    def load(cls, path: Path) -> "PipelineResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def drift_lines(self) -> list[str]:
        """End-to-end drift, one line per buffer: the worst case, and any inf/NaN disagreements."""
        worst: dict[str, tuple[float, str, dict]] = {}
        nonfinite: dict[str, list[str]] = {}
        for c in self.cases:
            for o in c.end_to_end:
                if o["tol_ratio"] >= worst.get(o["name"], (-1.0,))[0]:
                    worst[o["name"]] = (o["tol_ratio"], c.case, o)
                if o["nonfinite_mismatch"]:
                    nonfinite.setdefault(o["name"], []).append(f"{o['nonfinite_mismatch']} in {c.case}")
        lines = []
        for name, (ratio, case, o) in worst.items():
            line = (f"{name}: worst {ratio:.2g}x one kernel's {o['dtype']} tolerance (case {case}, "
                    f"max abs err {o['max_abs_err']:.3g})")
            if name in nonfinite:
                line += f"; inf/NaN where the float64 chain is finite (or the reverse): {', '.join(nonfinite[name])}"
            lines.append(line)
        return lines

    def failure_report(self, max_lines: int = 12) -> str:
        """Which stage failed first in each failing case, and how."""
        lines = []
        for c in self.cases:
            if c.passed:
                continue
            for s in c.stages:
                if s.passed:
                    continue
                if s.error:
                    lines.append(f"case {c.case}, stage {s.stage}: {s.error}")
                for o in s.outputs:
                    if not o["passed"]:
                        lines.append(f"case {c.case}, stage {s.stage}: {o['name']}: {o['reason']}")
                        if o.get("expected_slice"):
                            lines.append(f"    expected[{o['slice_start']}:] {o['expected_slice']}")
                            lines.append(f"    actual  [{o['slice_start']}:] {o['actual_slice']}")
                break  # later stages ran on data from a failing stage; the first failure is the one to fix
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... {len(lines) - max_lines} more lines"]
        return "\n".join(lines)


def drift(name: str, actual: torch.Tensor, exact: torch.Tensor, dtype: str, tol: Tolerance) -> dict:
    """How far the chain of ports ended from the float64 chain of references.

    ``tol_ratio`` is the worst ``|actual - exact| / (atol + rtol * |exact|)`` over elements finite in
    both: 1.0 means one kernel's tolerance used up. Relative error alone is useless here (values
    like probabilities can be 1e-90 in float64 and 0 in float32). Elements that are inf/NaN in one
    chain and not the same in the other are counted, not measured.
    """
    a = actual.detach().cpu().double().reshape(-1)
    e = exact.detach().cpu().double().reshape(-1)
    fin = torch.isfinite(a) & torch.isfinite(e)
    same_nonfinite = (torch.isnan(a) & torch.isnan(e)) | (torch.isinf(a) & torch.isinf(e) & (a.sign() == e.sign()))
    mismatch = int((~fin & ~same_nonfinite).sum())
    err = (a[fin] - e[fin]).abs()
    ratio = err / (tol.atol + tol.rtol * e[fin].abs()) if err.numel() else err
    return {"name": name, "dtype": dtype, "num_elements": int(a.numel()),
            "max_abs_err": float(err.max()) if err.numel() else 0.0,
            "tol_ratio": float(ratio.max()) if ratio.numel() else 0.0,
            "nonfinite_mismatch": mismatch,
            "within": mismatch == 0 and (float(ratio.max()) <= 1.0 if ratio.numel() else True)}


StageRunner = Callable[[ResolvedStage, CaseData], dict[str, torch.Tensor]]


def stage_inputs(pipe: PipelineSpec, stages: list[ResolvedStage], case: CaseSpec,
                 runner: StageRunner) -> list[CaseData]:
    """Each stage's real input for one case: the chain run once with ``runner``, inputs kept.

    Timing uses these so every stage is timed on the data it sees in the chain (sizes, and value
    ranges where they change the work, e.g. early exits on -inf).
    """
    bufs, _ = initial_buffers(pipe, case)
    datas = []
    for rs in stages:
        data = stage_data(rs, bufs, case.name, case.dims)
        datas.append(data)
        _store(rs, bufs, runner(rs, CaseData(data.case, data.dtype, data.dims,
                                             {k: v.clone() for k, v in data.tensors.items()},
                                             data.scalars, data.input_hash)))
    return datas


def run_pipeline(pipe: PipelineSpec, stages: list[ResolvedStage], runner: StageRunner, *, backend: str,
                 hardware: str, code_hashes: dict[str, str], reference_hashes: dict[str, str],
                 tol_overrides: dict[str, Tolerance] | None = None,
                 supported_dtypes: tuple[str, ...] | None = None) -> PipelineResult:
    """Run every case through the chain of ``runner`` calls and check each stage (see module doc)."""
    from kernelport.harness.verify import _short_error

    res = PipelineResult(pipeline=pipe.name, backend=backend, status="SKIPPED", hardware=hardware,
                         pipeline_hash=pipe.hash(), code_hashes=code_hashes, reference_hashes=reference_hashes)
    if supported_dtypes is not None:
        missing = sorted({rs.dtype for rs in stages} - set(supported_dtypes))
        if missing:
            res.reason = f"dtypes not supported by this backend: {', '.join(missing)}"
            return res
    not_runnable = []
    written = sorted({rs.bind[p.name] for rs in stages for p in rs.spec.params if p.is_output})
    for case in pipe.cases:
        start, h = initial_buffers(pipe, case)
        port_bufs = {k: v.clone() for k, v in start.items()}
        ref_bufs = {k: (v.double() if v.is_floating_point() else v.clone()) for k, v in start.items()}
        cr = PipelineCaseResult(case.name, False, h)
        broken = False
        for rs in stages:
            sr = StageResult(rs.label, rs.kernel_id, False)
            data = stage_data(rs, port_bufs, case.name, case.dims)
            try:
                expected = run_reference(rs.reference, rs.spec, data)
            except Exception as exc:
                sr.error = "reference failed: " + _short_error(exc)
                cr.stages.append(sr)
                broken = True
                break
            try:
                actual = runner(rs, data)
            except CaseNotRunnable as exc:
                cr.not_runnable = f"{case.name} stage {rs.label} ({exc})"
                break
            except Exception as exc:
                sr.error = _short_error(exc)
                cr.stages.append(sr)
                broken = True
                break
            ok = True
            for p in rs.spec.params:
                if not p.is_output:
                    continue
                out_dtype = p.resolved_dtype(rs.dtype)
                tol = tolerance_for(out_dtype, tol_overrides)
                if p.name not in actual:
                    sr.outputs.append({"name": p.name, "passed": False, "reason": "output missing"})
                    ok = False
                    continue
                tc = compare_tensor(p.name, actual[p.name], expected[p.name], TORCH_DTYPES[out_dtype], tol)
                sr.outputs.append(tc.to_dict())
                ok = ok and tc.passed
            sr.passed = ok
            cr.stages.append(sr)
            _store(rs, port_bufs, actual)
            # The float64 reference chain, for the end-to-end drift.
            _store(rs, ref_bufs, run_reference(rs.reference, rs.spec, stage_data(rs, ref_bufs, case.name, case.dims)))
        if cr.not_runnable:
            not_runnable.append(cr.not_runnable)
            continue
        cr.passed = not broken and len(cr.stages) == len(stages) and all(s.passed for s in cr.stages)
        if not broken:
            for b in written:
                dt = pipe.buffers[b].dtype
                cr.end_to_end.append(drift(b, port_bufs[b], ref_bufs[b], dt, tolerance_for(dt, tol_overrides)))
        res.cases.append(cr)
    notes = []
    if not_runnable:
        notes.append(f"cases this device cannot run, not tested: {'; '.join(not_runnable)}")
    res.reason = "; ".join(notes)
    if res.cases:
        res.status = "PASS" if all(c.passed for c in res.cases) else "FAIL"
    return res

