---
name: kernelport
description: Port CUDA kernels to Triton, Metal (MLX), WebGPU (WGSL), CPU (Numba) or HIP and PROVE each port correct with the kernelport MCP tools. Use when the user asks to port, convert, translate or verify CUDA kernels on other hardware.
---

# kernelport: port CUDA kernels and prove they work

You write the code. kernelport runs it on real hardware and checks it against a
trusted reference. **A port only counts as working when `verify_port` returns
`PASS`.** Nothing else (compiling, looking right, passing your own test) counts.

## Hard rules

1. The **original CUDA kernel is the ground truth for behaviour**. Never edit it.
2. The **reference is the source of truth for ports**. If the reference and the
   original disagree, fix the reference, never the original.
3. Never say a port works unless `verify_port` returned `PASS` for its current code.
   Report `FAIL`, `UNVERIFIED` and `SKIPPED` exactly as returned.
4. Before every retry, write one or two sentences explaining the bug you suspect,
   based on the failure report. Then fix and verify again.
5. Stop after **6 failed `verify_port` attempts** for a kernel/backend and report it as FAIL
   (the tool tells you `attempts_remaining`).
6. Do not try to change tolerances or inputs to make a port pass. Tolerances are
   fixed by kernelport. If you believe a tolerance or a test case is wrong, stop
   and explain to the user instead.
7. Keep ports readable: short comments mapping each part to the CUDA original.

## Workflow (call the tools in this order)

| Step | Do this | Tool |
|---|---|---|
| 0 | Check which backends can run here | `kernelport_doctor()` |
| 1 SCAN | Find kernels | `scan_kernels(path)` |
| 2 READ | Get source, parsed params, launch sites, suggested input spec | `read_kernel(kernel_id)` |
| 3 REFERENCE | Write a plain-PyTorch reference + complete input spec | `write_reference(kernel_id, code, spec)` |
| 4 VALIDATE | Run the ORIGINAL CUDA kernel vs your reference. PASS freezes it | `validate_reference(kernel_id)` |
| 5 TRANSLATE | Read the backend guide, write the port | `read_skill("<backend>")`, `write_port(kernel_id, backend, code)` |
| 6 VERIFY | Port vs frozen reference on every case | `verify_port(kernel_id, backend)` |
| 7 REPAIR | Explain suspected bug, fix, verify again (max 6 failures) | `write_port`, `verify_port` |
| 8 BENCHMARK | Only after PASS | `bench_port(kernel_id, backend)` |
| 8b TUNE (optional) | Python ports with a `TUNE` dict: verify + time each variant, keep the fastest passing one | `tune_port(kernel_id, backend)` |
| 9 REPORT | Show the user the status table and the report path | `list_status()`, `make_report()` |

Do the steps for one kernel at a time (validate it, port it, verify it) before
moving to the next; it keeps context small. Read `read_skill("common_bugs")` once
before your first port.

If `validate_reference` returns `UNAVAILABLE` (no NVIDIA GPU here), the reference
cannot be frozen on this machine. Still write the references and specs for every
kernel, then tell the user how to freeze them all at once on an NVIDIA machine:
commit `.kernelport/` and run `kernelport validate --all` there (a free Colab T4
works: `notebooks/freeze_references_colab.ipynb` in the kernelport repo). Once the
frozen `.kernelport/` is back in the repo, porting and verifying continue here.
Do not write ports for a kernel whose reference is not frozen: `verify_port` refuses.

If the target backend is not available here, still write the port; `verify_port`
returns `UNVERIFIED` and the port can be verified later with `kernelport verify` on
the right hardware. Say clearly that it is unverified.

## Step 3 in detail: the input spec (launch.json)

`read_kernel` returns `input_spec`: a suggestion with `"TODO"` wherever it could not
infer a value. Fill every TODO; `write_reference` rejects a spec with TODOs.

```json
{
  "name": "softmax",
  "source": "softmax.cu",
  "entry": "softmax<{T}>",
  "dtypes": ["float32", "float16", "bfloat16"],
  "params": [
    {"name": "x",    "kind": "input",  "dtype": "T", "shape": ["rows", "cols"]},
    {"name": "y",    "kind": "output", "dtype": "T", "shape": ["rows", "cols"]},
    {"name": "rows", "kind": "scalar", "dtype": "int32", "value": "rows"},
    {"name": "cols", "kind": "scalar", "dtype": "int32", "value": "cols"}
  ],
  "launch": {"grid": ["rows"], "block": [256], "shared_mem": 0},
  "cases": [
    {"name": "tiny_1x1",   "dims": {"rows": 1, "cols": 1}},
    {"name": "odd_5x1023", "dims": {"rows": 5, "cols": 1023}},
    {"name": "large_4096x4096", "dims": {"rows": 4096, "cols": 4096}}
  ],
  "init": {"x": {"dist": "normal", "std": 3.0, "edges": true}},
  "bench_case": "large_4096x4096"
}
```

- **params**: in the exact order of the CUDA signature. `kind` is `input`, `output`
  (kernel only writes it), `inout` (kernel reads and writes it, e.g. `y = a*x + y`,
  `atomicAdd(out, ...)`), or `scalar`. Check the suggested kinds against the source.
- **dtype**: `"T"` means "each entry of `dtypes`" (for templated kernels). Fixed types:
  `float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64`.
- **entry**: kernel name; for templates use `{T}` for the type, e.g. `"rmsnorm<{T}, 256>"`.
  Only list `dtypes` the original kernel actually supports (bfloat16 needs `__nv_bfloat16`
  instantiation to make sense).
- **source**: informational in a repo; kernelport compiles the file the kernel was
  scanned from (`file` in `read_kernel`), with host code stripped, so `main()`, cuBLAS
  calls and `#include <stdio.h>` in that file are fine.
- **shape / value / grid / block / shared_mem**: expressions over the case `dims`:
  `+ - * / // % **`, `cdiv(a, b)`, `min`, `max`, `next_pow2`. `grid`/`block` must reproduce
  exactly how the host code launches the kernel (see `launches` from `read_kernel`; it
  shows the expressions, so read the host code for the values). If the host picks the
  block size at run time (e.g. benchmarks a list of sizes), choose one it uses, 256 if
  present, and keep grid consistent with it.
- **dims** may be floats for float scalars: `{"n": 1000, "lr": 0.001, "eps": 1e-8}` with
  `{"name": "lr", "kind": "scalar", "dtype": "float32", "value": "lr"}`. Every reader of
  the case (original, reference, ports) gets the value rounded to the param's dtype, the
  way the kernel receives it (0.999 as float32 is 0.99900001...).
- **cases**: always include tiny (1 to 7 elements), odd sizes (1000, 1023, 4097),
  a power of two, and one large case. Use sizes that hit every code path (e.g. a row
  longer than one block, a dimension not divisible by the tile size). Keep every case
  inside the kernel's own limits: 32-bit `int` index math (`b*T*V` overflows past
  2^31), vectorised loads that need sizes divisible by 4 or 8, a block size that must
  not exceed some dimension. The reference runs on the CPU in float64, so a large case
  of ~50M elements is plenty. If the original fails only outside such a limit, say so
  in your report and keep the cases inside it; never edit the original.
- **init** must respect the kernel's input domain. The suggestion from `read_kernel` is
  `normal` + `edges` everywhere; change it where that is invalid: variances and second
  moments `>= 0` (`uniform`, `low: 0`), probabilities in (0, 1] before a `log`, indices
  with `randint` in range.
- **init** per input param: `dist` = `normal` (`mean`, `std`), `uniform` (`low`, `high`),
  `randint` (`low`, `high` exclusive, for index arrays: keep indices in range!),
  `zeros`, `ones`, `arange`. `edges: true` puts 0, -0, ±1, ±big, ±1/big first
  (`edge_big`, default 1000). Values may be expressions over dims: `"high": "vocab"`.
  Per-case overrides go in the case's own `"init"`.
- **step** (accumulating kernels: sums, matmuls, scans, atomics): add `"step": 0.0625`
  to every float input. Values become multiples of 1/16, so float32 sums are exact in
  any order and the comparison is not polluted by summation-order rounding.
- Outputs are pre-filled with NaN so unwritten elements are detected; inouts get
  generated values. Keep `"name"` values valid Python identifiers (`in` -> `in_`).

## Step 3 in detail: the reference

```python
import torch

def reference(x, rows, cols):          # names = launch.json param names
    return torch.softmax(x, dim=-1)     # one output: return the tensor
```

- Arguments: any subset of the `input`/`inout`/`scalar` params, by name.
- Float inputs arrive as **float64 CPU tensors** (integers keep their dtype). Write
  plain math; do not cast to the kernel dtype; kernelport rounds your result to the
  output dtype before comparing. Scalars arrive as Python numbers, already rounded to
  their declared dtype.
- Return one tensor (single output) or a dict `{output_name: tensor}`. For `inout`
  params return the **final** value (e.g. `{"y": a * x + y}`).
- Model what the kernel computes, including its quirks (e.g. biased vs unbiased
  variance, where eps is added, integer division, clamping).

`validate_reference` runs the original on every case. On FAIL, read the report,
fix the **reference** (or the spec: wrong kind, wrong shape, wrong grid), and retry.

## Step 6-7: reading a failure report

```
softmax/triton on <GPU>: FAIL (18/24 cases passed)
- case odd_3x4097 dtype float32:
  y shape [3, 4097]: 4368/12291 values outside tolerance ...; worst at [2, 3075]
  flat[11265:] expected [...]
  flat[11265:] actual   [...]
```

Use it: which cases fail (only wide rows? only fp16? only tiny?), where the worst
index is (tail of a row = masking or loop bounds; every element = formula), and
whether actual is NaN ("unexpected NaN" usually means an element was never written
or a 0/0). Errors are compact; an `error:` line means a crash or compile error with
the offending line of your port.

## Step 8b: tuning (optional)

Only worth it for kernels where speed matters, and only after PASS. Lift the knobs
you would tune (block size, warps, rows per program) to module-level constants and
list values to try in a literal `TUNE` dict (at most 32 combinations):

```python
BLOCK = 1024
NUM_WARPS = 4
TUNE = {"BLOCK": [256, 512, 1024, 2048], "NUM_WARPS": [4, 8]}
```

`write_port`, `verify_port` (the port must PASS as written), then `tune_port(kernel_id, backend)`.
Every variant is verified on every case before it is timed; a variant that FAILs is
just reported, it does not use your repair budget. If one is at least 3% faster, it
replaces the port, but only if each of its 5 timing rounds also beat each round of the
current port; otherwise the summary says the difference was within timing noise and the
port is kept. Report the result it returns, not your own estimate.

`bench_port` times port and original in 3 alternating runs and reports the median with
the range of the runs. On laptops and shared GPUs the runs can differ a lot; when the
summary says the runs differed by more than 20%, report the speedup as approximate (with
the range), or run `bench_port` again on a quieter machine.

## Step 9: what to tell the user

Call `make_report(format="markdown")` and show its `content` (the status matrix, benchmarks,
and what needs attention), then give the path of the HTML report from `make_report()`. For each kernel and
backend state PASS / FAIL / UNVERIFIED / SKIPPED exactly as reported, with the
hardware it was verified on and the speedup if benchmarked. For FAILs, include the
last suspected bug. Never round UNVERIFIED up to "should work".
