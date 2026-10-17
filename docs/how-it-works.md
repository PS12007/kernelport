# How kernelport proves a port correct

The agent writes code; kernelport decides whether it is right. Each step below exists to close
a specific way a port could look correct without being correct.

## 1. The reference is checked against the original, then frozen

The agent writes a plain-PyTorch `reference(...)` and a `launch.json` input spec (shapes,
dtypes, cases, how to launch the original). `validate_reference` compiles the **original CUDA
kernel** with NVRTC, runs it on every case and dtype, and compares it with the reference. Only
if every case passes is the pair copied into `.kernelport/kernels/<k>/frozen/` with a sha256
manifest. Any later edit to frozen files is detected, and `verify_port` refuses until the
reference is validated again.

This prevents a wrong reference (and a port matching it) from passing: the reference must agree
with the code being ported, not with the agent's reading of it.

## 2. Inputs are designed to expose bugs

- Specs cover tiny sizes (1 to 7 elements or rows, smaller than a block), odd sizes (1000, 1023,
  4097: tails and partial tiles), a power of two, and a large case (bigger than L2, used for
  timing). The skill requires this of every spec the agent writes; all zoo specs follow it.
- The first elements are edge values (0, -0, ±1, ±big, ±1/big); the rest are seeded, bounded
  normals or uniforms. Inputs are generated from PCG64 bits with exact IEEE operations only,
  so every platform regenerates bit-identical inputs (golden hashes are pinned in tests).
- **Outputs are pre-filled with NaN** (integer outputs with a sentinel), so an element a port
  never writes cannot accidentally hold the right value.
- Specs add targeted cases where a bug would otherwise slip through. Some found so far:
  - full-precision matmul cases (unquantised inputs, small K), because quantised inputs are exact
    in TF32 and a TF32 `tl.dot` port would have passed;
  - softmax rows wider than one block (4097), which catch reductions over only the first chunk;
  - a `reduce_sum` case whose accumulator starts non-zero, which catches ports that overwrite
    an `inout` instead of adding to it.

## 3. Comparison is strict and not negotiable

The reference runs in float64 on the CPU. Its result is rounded to the output dtype and compared
with `torch.testing`'s default tolerances for that dtype (float32: rtol 1.3e-6, atol 1e-5;
float16: rtol 1e-3; bfloat16: rtol 1.6e-2). **No tool accepts a tolerance.** A backend may
override a tolerance only in its config, with a written reason, and tests require that reason.

Reductions use inputs quantised to multiples of 1/16, so every summation order gives the exact
same float32 result and strict tolerances still apply to code that reorders sums.

## 4. Statuses are honest

| status | meaning |
|---|---|
| PASS | the current port code ran on real hardware and matched the frozen reference on every case |
| FAIL | at least one case mismatched or crashed; the report shows where and by how much |
| UNVERIFIED | the port exists but could not be run here, or it changed since its last result |
| SKIPPED | nothing to run (no port, or no dtype the backend supports) |

Results are stored per hardware with the port's and reference's hashes. When either changes,
the stored PASS becomes UNVERIFIED and benchmarks are marked stale. dtypes a backend cannot run
(e.g. bfloat16 on cpu) are skipped and **named** in the result. If none are left, the status
is SKIPPED, never PASS.

## 5. Repair is bounded

Each `verify_port` call is appended to `attempts.jsonl`. After 6 failed attempts since the last
PASS, the tool tells the agent to stop and report FAIL. Failure reports are compact (worst
element, a slice of expected vs actual, the port's own traceback lines), so they fit in an
agent's context.

## 6. Benchmarks only for verified code

`bench_port` refuses unless the current port code passed on this hardware. It times the port and
the original with device events (median of 50 launches after 10 warm-ups). CPU timings are
wall-clock and labelled not comparable.

## 7. Tuning only keeps verified, clearly faster code

`tune_port` (optional) works on Python ports that declare module-level constants and a literal
`TUNE` dict of values to try (at most 32 combinations). Each combination is rendered as a
variant file under `ports/<backend>/tune/` and **verified on every case first**, because a
block size can change results, not just speed. Passing variants are then timed in 5 interleaved
rounds, so GPU clock ramp-up and thermal drift affect them equally. A variant replaces the port
only if its median is at least 3% faster **and** each of its rounds beat each round of the
current configuration; the new port is then verified again so the stored PASS matches the new
code. Without the interleaving, a laptop GPU's clock ramp made a memory-bound softmax look
1.22x faster with a different block size; interleaved, every variant came within 2%, and the
port was left alone.

## 8. Pipelines check ports on the data they really get

A kernel's own cases are generated inputs. In a program, a kernel reads what the kernel before
it wrote: probabilities near 1/V after a softmax, normalized rows after a layernorm. A pipeline
(`.kernelport/pipelines/<name>.json`, saved with `write_pipeline`) runs several ported kernels in
a chain over named buffers, and **every stage is checked against its frozen reference on the
exact input that stage received**, with the usual tolerances. The first failing stage of each
case is reported.

It also runs the chain of references in float64 from the same starting buffers and reports how
far the chain of ports ended from it ("drift", as a multiple of one kernel's tolerance, plus a
count of values that are inf/NaN in one chain only). Drift is information, not a verdict:
rounding adds up over a chain, and the pipeline author never sets a tolerance.

The test suite has a crossentropy port that clamps probabilities to at least 1e-3 before the
log. It PASSes its kernel's own cases (probabilities drawn from [0.001, 1]) and FAILs the
softmax -> crossentropy pipeline. With backend `cuda`, the chain runs the original kernels, which
checks that they agree with their references on chained data too.

## The workspace

```
.kernelport/
  index.json                         kernel id -> slug
  kernels/<slug>/
    kernel.json                      analyzer snapshot
    launch.json, reference.py        candidate (editable)
    frozen/                          written only by a passing validate_reference
      launch.json reference.py manifest.json
    validation.json                  last validation result
    ports/<backend>/
      port.py                        the port
      attempts.jsonl                 every verify_port call
      results/<hardware>.json        latest result on each machine
      bench/<hardware>.json          latest timing on each machine
  pipelines/<name>.json              a chain of kernels (write_pipeline)
  pipelines/results/<name>/<backend>/<hardware>.json
  report.html, report.md
```

Every path is built from validated names and checked to stay inside `.kernelport/`. The tools
never write outside it, and the MCP server never takes the workspace root as an argument.
