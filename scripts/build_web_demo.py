"""Package WebGPU ports and test cases for the in-browser verifier (docs/web/).

    python scripts/build_web_demo.py

For every kernel with a WGSL port (the zoo, and the examples' frozen workspaces), and every float32
and float16 case small enough to download,
this runs the same harness ``kernelport verify`` uses: inputs from ``make_case``, expected
outputs from the frozen PyTorch reference (rounded to the output dtype), tolerances from
``tolerance_for``, the shader with the bindings the webgpu runner appends, the uniform bytes
and the workgroup counts from the port's ``grid()``. The page (docs/web/index.html) only
uploads, dispatches and compares, so a PASS in the browser means what a PASS means here.

Writes docs/web/cases.json and docs/web/data/<kernel>.bin. Needs torch, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelport import __version__  # noqa: E402
from kernelport.backends.base import bind_kwargs  # noqa: E402
from kernelport.backends.webgpu.runner import _ELEM, _SCALAR, _padded, bindings_source  # noqa: E402
from kernelport.harness.inputs import TORCH_DTYPES, make_case  # noqa: E402
from kernelport.harness.reference import load_module, run_reference  # noqa: E402
from kernelport.harness.spec import LaunchSpec  # noqa: E402
from kernelport.harness.tolerances import tolerance_for  # noqa: E402

OUT = ROOT / "docs" / "web"
MAX_CASE_BYTES = {"float32": 400_000, "float16": 150_000}  # inputs + expected outputs per case
DTYPES = ("float32", "float16")  # float16 cases run where the adapter has shader-f16


EXAMPLES = ("llm.c", "flash-attention-minimal")


def sources() -> list[dict]:
    """Every kernel with a WGSL port: the zoo, then the examples' frozen workspaces.

    Paths are repo-relative with forward slashes (they become GitHub links on the page).
    """
    found = []
    for kdir in sorted((ROOT / "zoo").iterdir()):
        if (kdir / "ports" / "webgpu.py").is_file():
            rel = f"zoo/{kdir.name}"
            found.append({"name": kdir.name, "group": "zoo", "port_file": f"{rel}/ports/webgpu.py",
                          "spec_file": f"{rel}/launch.json", "reference_file": f"{rel}/reference.py",
                          "cuda_file": f"{rel}/kernel.cu"})
    for ex in EXAMPLES:
        for kdir in sorted((ROOT / "examples" / ex / ".kernelport" / "kernels").iterdir()):
            if not (kdir / "ports" / "webgpu" / "port.py").is_file() or not (kdir / "frozen").is_dir():
                continue
            info = json.loads((kdir / "kernel.json").read_text(encoding="utf-8"))
            rel = f"examples/{ex}/.kernelport/kernels/{kdir.name}"
            found.append({"name": info["name"], "group": ex, "port_file": f"{rel}/ports/webgpu/port.py",
                          "spec_file": f"{rel}/frozen/launch.json", "reference_file": f"{rel}/frozen/reference.py",
                          "cuda_file": f"examples/{ex}/{info['file']}"})
    return found


def wgsl_for(spec: LaunchSpec, port, want: str) -> str:
    """The shader exactly as the webgpu runner builds it for this dtype."""
    overrides = dict(getattr(port, "BINDINGS", {}) or {})
    f16 = want == "float16" or any(p.resolved_dtype(want) == "float16" for p in spec.params if p.is_tensor)
    return ("enable f16; " if f16 else "") + port.WGSL + "\n" + bindings_source(spec, want, overrides)


def _kernel(src: dict, blob: bytearray, want: str) -> dict | None:
    spec = LaunchSpec.load(ROOT / src["spec_file"])
    if want not in spec.dtypes:
        return None
    reference = ROOT / src["reference_file"]
    port = load_module(ROOT / src["port_file"], "kp_web")
    code = wgsl_for(spec, port, want)
    cases = []
    for case, dtype in spec.iter_cases():
        if dtype != want:
            continue
        data = make_case(spec, case, dtype)
        expected = run_reference(reference, spec, data)
        arrays, buffers, outputs = {}, [], []
        size = 0
        for p in spec.params:
            if not p.is_tensor:
                continue
            np_t = _ELEM[p.resolved_dtype(dtype)][1]
            arr = data.tensors[p.name].contiguous().numpy().astype(np_t, copy=False)
            arrays[p.name] = arr
            raw = _padded(arr.tobytes())
            buffers.append({"name": p.name, "kind": "input" if p.kind == "input" else "rw",
                            "offset": len(blob), "size": len(raw)})
            blob += raw
            size += len(raw)
            if p.is_output:
                out_dtype = p.resolved_dtype(dtype)
                exp = expected[p.name].to(TORCH_DTYPES[out_dtype]).contiguous().numpy().astype(np_t, copy=False)
                tol = tolerance_for(out_dtype)
                outputs.append({"name": p.name, "binding": len(buffers) - 1, "dtype": out_dtype,
                                "count": int(exp.size), "offset": len(blob), "size": exp.nbytes,
                                "rtol": tol.rtol, "atol": tol.atol})
                blob += exp.tobytes()
                size += exp.nbytes
        if size > MAX_CASE_BYTES[want]:
            del blob[buffers[0]["offset"]:]  # drop this case's bytes again
            continue
        scalars = [p for p in spec.params if p.kind == "scalar"]
        uniform = _padded(b"".join(np.array(data.scalars[p.name], dtype=_SCALAR[p.dtype][1]).tobytes()
                                   for p in scalars), 16) if scalars else b""
        g = port.grid(**bind_kwargs(port.grid, spec, arrays, data.scalars))  # numpy, as the runner passes
        g = (g,) if isinstance(g, int) else tuple(int(x) for x in g)
        cases.append({"name": case.name, "dims": case.dims, "buffers": buffers, "outputs": outputs,
                      "uniform": list(uniform), "grid": list(g + (1,) * (3 - len(g)))})
    return {**src, "dtype": want, "wgsl": code, "cases": cases} if cases else None


def main() -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    kernels = []
    for dtype in DTYPES:
        for src in sources():
            blob = bytearray()
            k = _kernel(src, blob, dtype)
            if k is None:
                if dtype in LaunchSpec.load(ROOT / src["spec_file"]).dtypes:
                    print(f"{src['name']:32s} {dtype:8s} skipped: every case is over the size limit")
                continue
            stem = src["name"] if src["group"] == "zoo" else f"{src['group']}-{src['name']}"
            name = f"data/{stem}{'' if dtype == 'float32' else '-' + dtype}.bin"
            (OUT / name).write_bytes(bytes(blob))
            k["data"] = name
            k["data_sha256"] = hashlib.sha256(blob).hexdigest()
            kernels.append(k)
            print(f"{src['name']:32s} {dtype:8s} {len(k['cases'])} cases  {len(blob) / 1e6:5.2f} MB")
    manifest = {"kernelport": __version__, "dtypes": list(DTYPES), "kernels": kernels}
    (OUT / "cases.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8", newline="\n")
    total = sum((OUT / k["data"]).stat().st_size for k in kernels)
    print(f"wrote {OUT.relative_to(ROOT)}/cases.json + {len(kernels)} data files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
