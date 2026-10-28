"""Run a webgpu (WGSL) port through wgpu-native.

A webgpu port is a Python file defining::

    WGSL = '''
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < u32(params.n)) { y[i] = max(x[i], T(0.0)); }
    }
    '''

    def grid(n):                      # launch.json names (scalars, or arrays for .shape)
        return (cdiv(n, 256),)         # workgroups per dimension, each <= 65535

The runner appends the binding declarations, so the port never writes them:

- ``alias T = f32;`` (or ``f16`` for float16 cases; ``enable f16;`` is added)
- one storage buffer per tensor param, in launch.json order, named like the
  param: ``var<storage, read>`` for inputs, ``read_write`` for outputs/inouts.
  Element type: ``T`` for dtype "T", else f32 / f16 / i32 / u32; int64 arrives
  as ``vec2<u32>`` (low word in ``.x``), since core WGSL has no 64-bit integers.
- ``var<uniform> params: Params`` holding every scalar param (i32 / u32 / f32).

A port may override a buffer's declared type, e.g. for atomics:
``BINDINGS = {"out": "array<atomic<u32>>"}``.

Only core WebGPU features are enabled (plus shader-f16 when the adapter has it),
so a port that passes here uses WGSL a browser also accepts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from kernelport.backends.base import Prepared, bind_kwargs
from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import file_hash, load_module
from kernelport.harness.results import CaseNotRunnable
from kernelport.harness.spec import LaunchSpec

MAX_WORKGROUPS = 65535
PARAVIRTUAL_MAX_BOUND = 500 * 2**20
_ELEM = {"float32": ("f32", np.float32), "float16": ("f16", np.float16), "int32": ("i32", np.int32),
         "uint32": ("u32", np.uint32), "int64": ("vec2<u32>", np.int64)}
_SCALAR = {"int32": ("i32", np.int32), "int64": ("i32", np.int32), "uint32": ("u32", np.uint32),
           "float32": ("f32", np.float32), "float64": ("f32", np.float32)}

_device = None
_pipelines: dict[str, tuple] = {}
_loaded: dict[tuple[str, str], object] = {}


def device():
    """The shared wgpu device (high-performance adapter, core features + shader-f16)."""
    global _device
    if _device is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no WebGPU adapter found")
        lim = adapter.limits
        cap = 2**31 - 4  # what desktop browsers allow at most
        limits = {"max-storage-buffer-binding-size": min(lim["max-storage-buffer-binding-size"], cap),
                  "max-buffer-size": min(lim["max-buffer-size"], cap)}
        feats = ["shader-f16"] if "shader-f16" in adapter.features else []
        _device = adapter.request_device_sync(required_features=feats, required_limits=limits)
        _device._kp_adapter_info = adapter.info
    return _device


_fence = None


def sync() -> None:
    """Wait for all submitted work. Reading back a tiny buffer waits for the queue.

    (queue.on_submitted_work_done_sync is broken in wgpu 0.32: wrong C callback signature.)
    """
    global _fence
    import wgpu

    dev = device()
    if _fence is None:
        _fence = dev.create_buffer(size=4, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    dev.queue.read_buffer(_fence)


def has_f16() -> bool:
    return "shader-f16" in device().features


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def load_port(path: Path):
    """Import the port once per content hash; returns the module."""
    path = Path(path).resolve()
    key = (str(path), file_hash(path))
    if key not in _loaded:
        mod = load_module(path, "kp_wgpu")
        if not isinstance(getattr(mod, "WGSL", None), str):
            raise AttributeError(f"{path.name} must define WGSL = '''...''' (the compute shader)")
        if not callable(getattr(mod, "grid", None)):
            raise AttributeError(f"{path.name} must define grid(...) returning workgroup counts")
        _loaded[key] = mod
    return _loaded[key]


F32_MAX = float(np.finfo(np.float32).max)
F16_MAX = 65504.0
_FLOAT_LITERAL = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)([fh]?)(?![\w.])")


def check_browser_literals(wgsl: str) -> None:
    """Refuse float literals beyond the f32 (or f16) range.

    wgpu-native's compiler (naga) rounds ``-3.4028235e38`` to -FLT_MAX and accepts it,
    but browsers' compiler (Tint, in Chrome and Edge) rejects any literal above the
    largest finite f32, so the same port would fail in a browser.
    """
    code = re.sub(r"/\*.*?\*/", " ", re.sub(r"//[^\n]*", " ", wgsl), flags=re.S)
    for m in _FLOAT_LITERAL.finditer(code):
        limit = F16_MAX if m.group(2) == "h" else F32_MAX
        if abs(float(m.group(1))) > limit:
            line = code.count("\n", 0, m.start()) + 1
            raise ValueError(
                f"WGSL line {line}: the literal {m.group(0)} is larger than the largest finite "
                f"{'f16' if limit == F16_MAX else 'f32'} ({limit:.9g}). wgpu-native rounds it, but browsers (Tint) "
                "reject the shader. For -FLT_MAX write -3.4028234e38 (or bitcast<f32>(0xff7fffffu)).")


_LOOP_HEAD = re.compile(r"\b(?:for|while|loop)\b")
_BARE_VAR = re.compile(r"\bvar\s+([A-Za-z_]\w*)\s*:\s*([^=;{}]*?)\s*;")


def check_loop_vars(wgsl: str) -> None:
    """Refuse a `var` declared without an initializer inside a loop body.

    WGSL zero-initialises such a variable every time its declaration runs, i.e. on every
    iteration, and browsers do. naga before wgpu v30 zeroed it only once per invocation, so the
    value carried over between iterations (fixed upstream in gfx-rs/wgpu#9592; wgpu-py 0.32 still
    bundles wgpu-native 29.0.1.1, measured wrong on Vulkan, D3D12 and GL). Requiring an explicit
    initializer gives the same result on every version and in browsers.
    """
    code = re.sub(r"/\*.*?\*/", " ", re.sub(r"//[^\n]*", " ", wgsl), flags=re.S)
    for head in _LOOP_HEAD.finditer(code):
        open_ = code.find("{", head.end())
        if open_ < 0:
            continue
        depth, end = 0, len(code)
        for i in range(open_, len(code)):
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            if depth == 0:
                end = i
                break
        m = _BARE_VAR.search(code, open_, end)
        if m:
            line = code.count("\n", 0, m.start()) + 1
            raise ValueError(
                f"WGSL line {line}: `{m.group(0).strip()}` is declared inside a loop without an initializer. "
                "WGSL zeroes it on every iteration, but wgpu-native before v30 (naga) only once, so results "
                f"differ between versions and browsers. Initialise it explicitly: `var {m.group(1)} = "
                f"{m.group(2)}();` (the zero value).")


_WGSL_BARRIERS = re.compile(r"\b(workgroupBarrier|storageBarrier|textureBarrier|workgroupUniformLoad)\s*\(")
_WGSL_THREAD_BUILTIN = re.compile(
    r"@builtin\s*\(\s*(?:local_invocation_id|global_invocation_id|local_invocation_index"
    r"|subgroup_invocation_id)\s*\)\s*([A-Za-z_]\w*)\s*:")
_WGSL_FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*\(")


def _wgsl_as_c(code: str) -> str:
    """Rewrite WGSL so the C-style hazard analysis reads it: parenthesise every ``if``/``while``
    condition (WGSL allows ``if i >= n {``) and drop type annotations from declarations.
    No newlines are added or removed, so line numbers stay put."""
    def paren(m: re.Match) -> str:
        cond = m.group(2).strip()
        if cond.startswith("(") and _matching_paren(cond, 0) == len(cond):
            return f"{m.group(1)} {cond} {{"
        return f"{m.group(1)} ({cond}) {{"

    code = re.sub(r"\b(if|while)\b([^{};]*)\{", paren, code)
    return re.sub(r"\b(let|var|const)\s+([A-Za-z_]\w*)\s*:\s*[^=;\n]+=", r"\1 \2 =", code)


def check_barrier_uniformity(wgsl: str) -> None:
    """Refuse a barrier that some invocations of the workgroup can skip.

    WGSL requires ``workgroupBarrier()`` (and ``storageBarrier``, ``workgroupUniformLoad``) in
    uniform control flow. Browsers (Tint, in Chrome and Edge) reject a barrier after an
    ``if (gid.x >= n) { return; }`` or inside a branch or loop that depends on the invocation id,
    while wgpu-native 29 (naga) accepts all of these, so such a port would PASS here and fail to
    compile in a browser. CUDA originals often return early before ``__syncthreads()`` (fine on
    NVIDIA), which makes this an easy thing to copy into a port. Only values derived from the
    invocation-id builtins count as non-uniform; Tint also treats loads from read_write storage
    as non-uniform, which this check does not follow.
    """
    from kernelport.analyzer.hazards import divergent_barriers

    code = _wgsl_as_c(re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                             re.sub(r"//[^\n]*", " ", wgsl), flags=re.S))
    funcs: dict[str, tuple[int, str, str]] = {}  # name -> (body offset, params, body)
    for m in _WGSL_FN.finditer(code):
        close = _matching_paren(code, m.end() - 1)
        start = code.find("{", close) if close > 0 else -1
        if start < 0:
            continue
        end = _matching_brace(code, start)
        funcs[m.group(1)] = (start, code[m.end():close], code[start:end])
    helpers = {n: b for n, (_, _, b) in funcs.items()}
    for name, (start, params, body) in funcs.items():
        ids = _WGSL_THREAD_BUILTIN.findall(params)
        if not ids:
            continue
        sources = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, ids)) + r")\b")
        others = {n: b for n, b in helpers.items() if n != name}
        hits = divergent_barriers(body, others, sources=sources, barriers=_WGSL_BARRIERS, strict=True)
        if not hits:
            continue
        kind, at, cond, op, bar = hits[0]
        line = code.count("\n", 0, start + at) + 1
        bar_line = code.count("\n", 0, start + op) + 1
        what = {"exit": f"`if ({cond})` lets some invocations return before `{bar}()` at line {bar_line}",
                "branch": f"`{bar}()` at line {bar_line} is inside `if ({cond})`",
                "loop": f"`{bar}()` at line {bar_line} is inside a loop whose condition `{cond}` "
                        "depends on the invocation"}[kind]
        raise ValueError(
            f"WGSL line {line}: {what}. Barriers must be in uniform control flow: wgpu-native accepts this, "
            "but browsers (Tint) reject the shader ('workgroupBarrier' must only be called from uniform "
            "control flow). Keep every invocation running to the barrier and guard the work instead, e.g. "
            "`if (i < n) { ... }` around the loads and stores, with the barrier outside it.")


def _matching_paren(code: str, i: int) -> int:
    depth = 0
    for j in range(i, len(code)):
        depth += {"(": 1, ")": -1}.get(code[j], 0)
        if depth == 0:
            return j + 1
    return -1


def _matching_brace(code: str, i: int) -> int:
    depth = 0
    for j in range(i, len(code)):
        depth += {"{": 1, "}": -1}.get(code[j], 0)
        if depth == 0:
            return j + 1
    return len(code)


def _elem_type(p, case_dtype: str) -> tuple[str, type]:
    dt = p.resolved_dtype(case_dtype)
    if dt not in _ELEM:
        raise TypeError(f"{p.name}: dtype {dt} is not supported by the webgpu backend")
    wgsl, np_t = _ELEM[dt]
    return ("T" if p.dtype == "T" else wgsl), np_t


def bindings_source(spec: LaunchSpec, case_dtype: str, overrides: dict[str, str]) -> str:
    """WGSL declarations the runner appends to the port's shader."""
    t = "f16" if case_dtype == "float16" else "f32"
    lines = [f"alias T = {t};"]
    b = 0
    for p in spec.params:
        if not p.is_tensor:
            continue
        elem, _ = _elem_type(p, case_dtype)
        access = "read" if p.kind == "input" else "read_write"
        ty = overrides.get(p.name, f"array<{elem}>")
        lines.append(f"@group(0) @binding({b}) var<storage, {access}> {p.name}: {ty};")
        b += 1
    scalars = [p for p in spec.params if p.kind == "scalar"]
    if scalars:
        fields = ", ".join(f"{p.name}: {_SCALAR[p.dtype][0]}" for p in scalars)
        lines.append(f"struct Params {{ {fields} }};")
        lines.append(f"@group(0) @binding({b}) var<uniform> params: Params;")
    return "\n".join(lines) + "\n"


def _pipeline(code: str, kinds: list[str]):
    """Compile (cached) shader + explicit bind group layout."""
    import wgpu

    key = code + "|" + ",".join(kinds)
    if key not in _pipelines:
        dev = device()
        types = {"input": wgpu.BufferBindingType.read_only_storage,
                 "rw": wgpu.BufferBindingType.storage,
                 "uniform": wgpu.BufferBindingType.uniform}
        bgl = dev.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": types[k]}}
            for i, k in enumerate(kinds)])
        layout = dev.create_pipeline_layout(bind_group_layouts=[bgl])
        module = dev.create_shader_module(code=code)
        pipe = dev.create_compute_pipeline(layout=layout, compute={"module": module, "entry_point": "main"})
        _pipelines[key] = (pipe, bgl)
    return _pipelines[key]


def _padded(raw: bytes, multiple: int = 4) -> bytes:
    extra = (-len(raw)) % multiple
    return raw + b"\0" * extra if extra or not raw else raw


def binding_limits() -> tuple[int, int | None]:
    """(largest single storage binding, largest total bound per dispatch or None) in bytes."""
    dev = device()
    per = min(dev.limits["max-storage-buffer-binding-size"], dev.limits["max-buffer-size"])
    total = None
    if "Paravirtual" in str(dev._kp_adapter_info.get("device", "")):
        # Apple's paravirtual GPU (macOS VMs such as GitHub's macos-14 runners) reports 3.5 GiB
        # per binding, but a dispatch binding 628 MiB in total silently does not run (outputs
        # untouched), while 500 MiB was measured to work. Stay within what was measured.
        per = total = min(per, PARAVIRTUAL_MAX_BOUND)
    return per, total


def _check_binding_sizes(spec: LaunchSpec, data: CaseData, per: int, total: int | None) -> None:
    """Raise CaseNotRunnable if the case's tensors exceed what this adapter can bind."""
    sizes = {}
    for p in spec.params:
        if p.is_tensor:
            t = data.tensors[p.name]
            itemsize = np.dtype(_elem_type(p, data.dtype)[1]).itemsize if t.dtype != torch.bfloat16 else 2
            sizes[p.name] = t.numel() * itemsize
    mib = 2**20
    for name, n in sizes.items():
        if n > per:
            raise CaseNotRunnable(f"{name} needs a {n / mib:.0f} MiB storage binding; "
                                  f"this adapter allows {per / mib:.0f} MiB")
    if total is not None and sum(sizes.values()) > total:
        raise CaseNotRunnable(f"the case binds {sum(sizes.values()) / mib:.0f} MiB in total; "
                              f"this adapter runs at most {total / mib:.0f} MiB per dispatch")


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Compile the port for this case's dtype and upload the case's buffers."""
    import wgpu

    mod = load_port(path)
    dev = device()
    enable = "enable f16; " if data.dtype == "float16" or any(
        p.resolved_dtype(data.dtype) == "float16" for p in spec.params if p.is_tensor) else ""
    if enable and not has_f16():
        raise TypeError("this WebGPU adapter has no shader-f16 support")
    overrides = dict(getattr(mod, "BINDINGS", {}) or {})
    unknown = set(overrides) - {p.name for p in spec.params if p.is_tensor}
    if unknown:
        raise KeyError(f"BINDINGS names unknown tensor params {sorted(unknown)}")
    check_browser_literals(mod.WGSL)
    check_loop_vars(mod.WGSL)
    check_barrier_uniformity(mod.WGSL)
    code = enable + mod.WGSL + "\n" + bindings_source(spec, data.dtype, overrides)

    _check_binding_sizes(spec, data, *binding_limits())

    kinds, buffers, outputs = [], [], {}
    arrays = {}
    for p in spec.params:
        if not p.is_tensor:
            continue
        t = data.tensors[p.name]
        if t.dtype == torch.bfloat16:
            raise TypeError(f"{p.name}: bfloat16 is not a WGSL type; the webgpu backend skips bfloat16 cases")
        _, np_t = _elem_type(p, data.dtype)
        arr = t.contiguous().numpy().astype(np_t, copy=False)
        arrays[p.name] = arr
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        buf = dev.create_buffer_with_data(data=_padded(arr.tobytes()), usage=usage)
        buffers.append(buf)
        kinds.append("input" if p.kind == "input" else "rw")
        if p.is_output:
            outputs[p.name] = (buf, arr)
    scalars = [p for p in spec.params if p.kind == "scalar"]
    if scalars:
        vals = []
        for p in scalars:
            v = data.scalars[p.name]
            if p.dtype in ("int64", "int32") and not (-2**31 <= int(v) < 2**31):
                raise OverflowError(f"scalar {p.name}={v} does not fit in i32")
            vals.append(np.array(v, dtype=_SCALAR[p.dtype][1]).tobytes())
        ubuf = dev.create_buffer_with_data(data=_padded(b"".join(vals), 16), usage=wgpu.BufferUsage.UNIFORM)
        buffers.append(ubuf)
        kinds.append("uniform")

    pipe, bgl = _pipeline(code, kinds)
    group = dev.create_bind_group(layout=bgl, entries=[
        {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(buffers)])

    kwargs = bind_kwargs(mod.grid, spec, arrays, data.scalars)
    g = mod.grid(**kwargs)
    g = (g,) if isinstance(g, int) else tuple(int(x) for x in g)
    if not 1 <= len(g) <= 3 or any(x < 0 for x in g):
        raise ValueError(f"grid() must return 1 to 3 non-negative ints, got {g}")
    if any(x > MAX_WORKGROUPS for x in g):
        raise ValueError(f"grid() returned {g}: WebGPU allows at most {MAX_WORKGROUPS} workgroups per "
                         "dimension; use a 2-D grid or loop inside the shader")
    g = g + (1,) * (3 - len(g))

    def launch() -> None:
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups(*g)
        cp.end()
        dev.queue.submit([enc.finish()])

    def fetch() -> dict:
        out = {}
        for name, (buf, arr) in outputs.items():
            raw = dev.queue.read_buffer(buf)
            got = np.frombuffer(raw, dtype=arr.dtype, count=arr.size).reshape(arr.shape).copy()
            out[name] = torch.from_numpy(got)
        return out

    return Prepared(launch=launch, sync=sync, fetch=fetch, timer="wall_clock")


def adapter_name() -> str:
    info = device()._kp_adapter_info
    name = info.get("device") or "WebGPU adapter"
    return f"{name} (WebGPU/{info.get('backend_type', '?')})"

