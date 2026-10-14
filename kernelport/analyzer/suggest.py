"""Draft a launch.json for a parsed kernel.

The draft is a starting point for the agent: kinds and dtypes come from the
signature and body, but tensor shapes, case sizes and the grid usually need a
human or agent to fill in. Unknowns are marked with "TODO" so validation fails
loudly until they are resolved.
"""

from __future__ import annotations

import keyword
import re

from kernelport.analyzer.models import KernelInfo

TODO = "TODO"
STANDARD_1D_SIZES = [("tiny_1", 1), ("tiny_7", 7), ("odd_1000", 1000), ("odd_1023", 1023),
                     ("odd_4097", 4097), ("pow2_65536", 65536), ("large_16M", 16_777_216)]


def safe_name(name: str) -> str:
    """Python-safe parameter name (``in`` -> ``in_``)."""
    name = re.sub(r"\W", "_", name) or "arg"
    return name + "_" if keyword.iskeyword(name) or name[0].isdigit() else name


def _dtypes_for(kernel: KernelInfo) -> list[str]:
    if any(p.dtype == "T" for p in kernel.params):
        return ["float32", "float16"]
    return ["float32"]


def _entry_for(kernel: KernelInfo) -> str:
    if not kernel.template_params:
        return kernel.name
    # Only the first type parameter maps to {T}; other template args need filling in.
    args = []
    first_type_done = False
    for tp in kernel.template_params:
        if re.match(r"(?:typename|class)\s", tp) and not first_type_done:
            args.append("{T}")
            first_type_done = True
        else:
            default = tp.split("=", 1)[1].strip() if "=" in tp else TODO
            args.append(default)
    return f"{kernel.name}<{', '.join(args)}>"


def suggest_spec(kernel: KernelInfo) -> dict:
    """Return a draft launch.json dict for ``kernel``."""
    scalar_names = [safe_name(p.name) for p in kernel.params if not p.is_pointer]
    one_dim = len(scalar_names) >= 1 and sum(p.is_pointer for p in kernel.params) >= 1
    size_dim = next((n for n in scalar_names if n.lower() in ("n", "size", "numel", "num", "len", "length")), None)

    params = []
    for p in kernel.params:
        name = safe_name(p.name)
        entry: dict = {"name": name, "kind": p.kind, "dtype": p.dtype or TODO}
        if p.is_pointer:
            entry["shape"] = [size_dim] if size_dim else [TODO]
        else:
            entry["value"] = name  # scalars come from case dims of the same name
        if name != p.name:
            entry["cuda_name"] = p.name
        params.append(entry)

    launch = {"grid": [TODO], "block": [TODO], "shared_mem": 0}
    notes = []
    if kernel.launches:
        l0 = kernel.launches[0]
        notes.append(f"launch site {l0.file}:{l0.line}: <<<{l0.grid}, {l0.block}"
                     + (f", {l0.shared_mem}" if l0.shared_mem not in ("", "0") else "") + ">>>"
                     + (f" args ({', '.join(l0.args)})" if l0.args else ""))
        if re.fullmatch(r"\d+", l0.block or ""):
            launch["block"] = [int(l0.block)]
    if "dynamic_shared_memory" in kernel.features:
        launch["shared_mem"] = TODO
        notes.append("kernel uses extern __shared__: set shared_mem bytes")

    if size_dim and one_dim:
        cases = [{"name": n, "dims": {size_dim: v}} for n, v in STANDARD_1D_SIZES]
    else:
        cases = [{"name": "tiny", "dims": {d: TODO for d in scalar_names}},
                 {"name": "odd", "dims": {d: TODO for d in scalar_names}},
                 {"name": "pow2", "dims": {d: TODO for d in scalar_names}},
                 {"name": "large", "dims": {d: TODO for d in scalar_names}}]
    init = {safe_name(p.name): {"dist": "normal", "edges": True}
            for p in kernel.params if p.is_pointer and p.kind != "output"
            and (p.dtype in ("T", "float32", "float16", "bfloat16", "float64"))}
    for p in kernel.params:
        if p.is_pointer and p.kind != "output" and p.dtype and p.dtype.startswith("int"):
            init[safe_name(p.name)] = {"dist": "randint", "low": 0, "high": TODO}
            notes.append(f"{p.name} looks like an index array: set its valid range")
    if any(f in kernel.features for f in ("atomics",)) or any(
            k in kernel.name.lower() for k in ("sum", "reduce", "matmul", "gemm", "scan", "cumsum", "dot")):
        notes.append("accumulating kernel: consider \"step\": 0.0625 in init so float sums are exact in any order")
    if any(p.dtype == "T" for p in kernel.params):
        notes.append("templated on T: add \"bfloat16\" to dtypes if the source includes cuda_bf16.h "
                     "and T may be __nv_bfloat16")

    return {
        "name": kernel.name,
        "source": kernel.file,
        "entry": _entry_for(kernel),
        "dtypes": _dtypes_for(kernel),
        "params": params,
        "launch": launch,
        "cases": cases,
        "init": init,
        "_notes": notes,
    }


def has_todo(obj) -> bool:
    """True if any value in a (nested) spec dict is still TODO."""
    if isinstance(obj, dict):
        return any(has_todo(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_todo(v) for v in obj)
    return obj == TODO
