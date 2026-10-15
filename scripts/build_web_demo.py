"""Package the zoo's WebGPU ports and test cases for the in-browser verifier (docs/web/).

    python scripts/build_web_demo.py

For every zoo kernel with a WGSL port, and every float32 case small enough to download,
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
import torch

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
MAX_CASE_BYTES = 400_000  # inputs + expected outputs per case; keeps the page download small
DTYPE = "float32"


def _kernel(kdir: Path, blob: bytearray) -> dict | None:
    port_path = kdir / "ports" / "webgpu.py"
    if not port_path.is_file():
        return None
    spec = LaunchSpec.load(kdir / "launch.json")
    if DTYPE not in spec.dtypes:
        return None
    port = load_module(port_path, "kp_web")
    overrides = dict(getattr(port, "BINDINGS", {}) or {})
    code = port.WGSL + "\n" + bindings_source(spec, DTYPE, overrides)
    cases = []
    for case, dtype in spec.iter_cases():
        if dtype != DTYPE:
            continue
        data = make_case(spec, case, dtype)
        expected = run_reference(kdir / "reference.py", spec, data)
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
        if size > MAX_CASE_BYTES:
            del blob[buffers[0]["offset"]:]  # drop this case's bytes again
            continue
        scalars = [p for p in spec.params if p.kind == "scalar"]
        uniform = _padded(b"".join(np.array(data.scalars[p.name], dtype=_SCALAR[p.dtype][1]).tobytes()
                                   for p in scalars), 16) if scalars else b""
        g = port.grid(**bind_kwargs(port.grid, spec, {k: torch.from_numpy(v) for k, v in arrays.items()},
                                    data.scalars))
        g = (g,) if isinstance(g, int) else tuple(int(x) for x in g)
        cases.append({"name": case.name, "dims": case.dims, "buffers": buffers, "outputs": outputs,
                      "uniform": list(uniform), "grid": list(g + (1,) * (3 - len(g)))})
    return {"name": kdir.name, "wgsl": code, "port_file": f"zoo/{kdir.name}/ports/webgpu.py",
            "cuda_file": f"zoo/{kdir.name}/kernel.cu", "cases": cases} if cases else None


def main() -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    kernels = []
    for kdir in sorted((ROOT / "zoo").iterdir()):
        blob = bytearray()
        k = _kernel(kdir, blob)
        if k is None:
            continue
        name = f"data/{kdir.name}.bin"
        (OUT / name).write_bytes(bytes(blob))
        k["data"] = name
        k["data_sha256"] = hashlib.sha256(blob).hexdigest()
        kernels.append(k)
        print(f"{kdir.name:18s} {len(k['cases'])} cases  {len(blob) / 1e6:5.2f} MB")
    manifest = {"kernelport": __version__, "dtype": DTYPE, "kernels": kernels}
    (OUT / "cases.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8", newline="\n")
    total = sum((OUT / k["data"]).stat().st_size for k in kernels)
    print(f"wrote {OUT.relative_to(ROOT)}/cases.json + {len(kernels)} data files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
