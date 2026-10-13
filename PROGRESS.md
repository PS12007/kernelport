# PROGRESS

## Done
- **M0 Environment** (2026-10-13): `kernelport doctor` / `scripts/check_env.py` pass on Windows 11,
  RTX 5060 Laptop (sm_120). Python 3.12.10, torch 2.11.0+cu128, triton-windows 3.7.1.post27,
  cupy-cuda12x 14.2.0 (NVRTC from `nvidia-cuda-nvrtc-cu12` pip wheel; no nvcc / Visual Studio),
  numba 0.68.0. Smoke kernels pass for cuda (original), triton, cpu. metal/webgpu/hip skipped
  with reasons.

- **M1 Harness** (2026-10-13): launch spec (`launch.json`), deterministic inputs, CuPy/NVRTC runner for
  originals, reference runner (float64), compare, result schema, CUDA-event timer, `kernelport zoo`.
  vector_add / relu / softmax: originals PASS vs reference, hand-written Triton ports PASS on all
  cases (21/21/24 cases across fp32/fp16/bf16). Speedup vs original on the bench case: 1.00x / 1.01x / 0.99x
  (all memory bound, ~300 GB/s). 8 deliberately broken ports (missing tail, wrong op, 5e-6 relative error,
  compile error, no-op, leaky relu, no max-subtraction, partial row sum) all FAIL with the right reason.
  62 tests pass.

- **M2 Zoo** (2026-10-13): all 12 kernels (vector_add, relu, saxpy, transpose, reduce_sum, softmax,
  layernorm, rmsnorm, matmul_naive, rope, embedding_gather, cumsum) have kernel.cu + launch.json +
  reference.py, and every original PASSES its reference on every case (fp32 everywhere, fp16 everywhere,
  bf16 where the kernel supports it). Kernels cover templates, shared memory tiles, __syncthreads,
  warp shuffles, atomics, block reductions, scans, gathers, int64 indices, in-place (inout) params.
  Mutation test: for each kernel a subtly wrong reference (dropped element, unbiased variance, exclusive
  scan, wrong rotation sign, clamped ids, ...) is REJECTED by validation.

- **M3 Analyzer** (2026-10-13): `kernelport/analyzer/` (pure Python, no torch). Finds `__global__`
  definitions in .cu/.cuh/.h/.cpp and in CUDA strings inside .py files (CuPy RawKernel style); skips
  comments, strings, declarations and __device__ functions. Parses templates (with defaults),
  __launch_bounds__, extern "C", multi-line params; infers each pointer's kind (input/output/inout)
  from body usage, following local pointer aliases, atomics, derefs and calls; detects 17 features
  (shared mem, syncthreads, shuffles, atomics, fp16/bf16, vector loads, tensor cores, inline PTX, ...);
  finds `<<<...>>>` launch sites. `suggest_spec` drafts a launch.json with TODOs. Agrees with all 12
  hand-written zoo specs on every param kind and dtype. `kernelport scan <path>`.

- **M4 Backends** (2026-10-13): `kernelport/backends/base.py` (BackendConfig, Backend facade, Prepared,
  registry). Each backend = config.py + capability.py (+ runner.py once implemented). Triton moved into
  `backends/triton/`; cpu/metal/webgpu/hip registered as stubs whose capability checks give clear reasons.
  Verification skips dtypes a backend does not support and names them in the result; if nothing is left
  the status is SKIPPED (never PASS). Doctor reads port backends from the registry. Test proves core
  imports (CLI, analyzer, spec, registry, configs) pull in no torch/GPU package; wheel's base
  Requires-Dist is only mcp, typer, rich, numpy. 119 tests pass.

## Next
- M5 Workspace (.kernelport/) with frozen references, input specs, attempt logs, results.

## Known issues / notes
- The zoo is not shipped in the wheel; `kernelport zoo` needs a repo checkout (or cwd containing zoo/).
- Analyzer is lexical, not a C++ front end: macros that generate kernels, kernels behind #if branches, and
  pointer kinds through complex aliasing (structs of pointers, pointer arithmetic in loops) may be wrong.
  Kinds are suggestions; the agent confirms them, and reference validation catches mistakes.
- Reductions (reduce_sum, matmul_naive, cumsum) use inputs quantized to multiples of 1/16 (`"step"` in
  launch.json init). With unquantized normals, the fp32 originals FAILED the torch.testing fp32 tolerance
  purely from summation-order rounding and cancellation (e.g. 1000 + 0.001 - 1000). Quantized inputs make
  every summation order exact, so strict tolerances stay and real bugs still show. Tolerances were NOT
  changed. Alternative for later (needs owner sign-off): a condition-aware tolerance for accumulations.
- Harness design decisions (M1):
  - Inputs come from PCG64 `random_raw` bits converted with exact IEEE ops only (no log/sin/cos), so every
    platform regenerates bit-identical inputs. Golden hashes in tests/test_inputs.py pin this.
    "normal" is Irwin-Hall (sum of 4 uniforms), mean 0 std 1, bounded to about +-3.46 std.
  - Output buffers are pre-filled with NaN (ints: a sentinel) so unwritten elements are caught.
  - References run on CPU in float64; the result is rounded to the output dtype, then compared with
    torch.testing default tolerances (see kernelport/harness/tolerances.py).
  - NVRTC lacks host math macros; a small prelude defines INFINITY/NAN if missing (`#line 1` keeps
    error line numbers correct).
- Timer: per-launch CUDA events, median of 50 after 10 warmups; no L2 flush (bench cases are larger than L2).
- `cupy-cuda12x[ctk]` is intentionally NOT used: it pulls ~1 GB of cuBLAS/cuFFT/cuSOLVER we do not need.
  Only NVRTC + CUDA runtime wheels are installed.
- torch CUDA must be installed from the PyTorch index (`--index-url https://download.pytorch.org/whl/cu128`)
  before `pip install kernelport[cuda,triton]`; extras cannot select the CUDA build of torch.
- triton-windows minor version must match torch: torch 2.11 -> triton-windows 3.7.x.
