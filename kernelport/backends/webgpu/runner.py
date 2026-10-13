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

from pathlib import Path

import numpy as np
import torch

from kernelport.backends.base import Prepared, bind_kwargs
from kernelport.harness.inputs import CaseData
from kernelport.harness.reference import file_hash, load_module
from kernelport.harness.spec import LaunchSpec

MAX_WORKGROUPS = 65535
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


def _elem_type(p, case_dtype: str) -> tuple[str, type]:
    dt = p.resolved_dtype(case_dtype)
    if dt not in _ELEM:
        raise TypeError(f"{p.name}: dtype {dt} is not supported by the webgpu backend")
    if dt == "float16" and not has_f16():
        raise TypeError("this WebGPU adapter has no shader-f16 support")
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


def prepare(path: Path, spec: LaunchSpec, data: CaseData) -> Prepared:
    """Compile the port for this case's dtype and upload the case's buffers."""
    import wgpu

    mod = load_port(path)
    dev = device()
    enable = "enable f16; " if data.dtype == "float16" or any(
        p.resolved_dtype(data.dtype) == "float16" for p in spec.params if p.is_tensor) else ""
    overrides = dict(getattr(mod, "BINDINGS", {}) or {})
    unknown = set(overrides) - {p.name for p in spec.params if p.is_tensor}
    if unknown:
        raise KeyError(f"BINDINGS names unknown tensor params {sorted(unknown)}")
    code = enable + mod.WGSL + "\n" + bindings_source(spec, data.dtype, overrides)

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

