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

- **M5 Workspace** (2026-10-13): `kernelport/workspace.py`. `.kernelport/` holds index, analyzer snapshots,
  candidate spec + reference, `frozen/` (written only by a passing CUDA validation, with a sha256 manifest;
  any edit to frozen files is detected and blocks verification), ports per backend, `attempts.jsonl`, and
  latest result per hardware. Every path is built from validated names and checked to stay inside
  `.kernelport/` (no `..`, separators, drive letters, or symlink escapes). Status is honest: a stored PASS
  becomes UNVERIFIED if the port or frozen reference changed since. Files are written byte-exact so hashes
  match on Windows. 29 workspace tests (1 skipped: symlinks need admin on Windows).

- **M6 MCP server + CLI** (2026-10-13): `kernelport/tools.py` implements the 11 tools once
  (kernelport_doctor, scan_kernels, read_kernel, read_skill, write_reference, validate_reference,
  write_port, verify_port, bench_port, list_status, make_report); `kernelport/mcp_server.py` exposes them
  over stdio with the official MCP Python SDK (2.x `MCPServer`; its stdio transport diverts fd 1 to
  stderr so GPU-library prints cannot corrupt the protocol). The workspace root is never a tool argument.
  Every tool has a CLI command (scan, read, skill, write-reference, validate, write-port, verify, bench,
  status, report, mcp). Enforced in tools: verify refuses unfrozen/tampered references, bench refuses
  unless the current code PASSED on this hardware, no tool takes tolerances, attempts logged, repair
  budget reported. Tests call every tool directly, run the full workflow on the GPU, and talk to the real
  server over stdio. 161 tests pass.

- **M7 Skills** (2026-10-13): `skills/kernelport/SKILL.md` (frontmatter, hard rules, 9-step workflow with
  tool per step, launch.json and reference format, reading failure reports, final reporting),
  `backends/triton.md` (CUDA->Triton mapping table, patterns, precision notes, checklist; every Triton API
  named was checked against triton 3.7.1), `common_bugs.md` (symptom -> cause table, checklist).
  Found while writing the guide: quantized matmul inputs are exact in TF32, so a TF32 port would have
  passed. Added two full-precision matmul cases (small K); a TF32 `tl.dot` port now FAILS exactly those
  cases while the `input_precision="ieee"` port passes 16/16. Skill-consistency tests (every tool and
  param kind documented). 168 tests pass.

- **M8 Setup** (2026-10-13): `kernelport setup [--uninstall] [--print] [--agent X] [--yes]` for Claude Code,
  Codex, Cursor, Gemini CLI (`kernelport/setup_agents/`). Locations from each agent's official docs
  (see detect.py docstring). Shows exact changes, asks approval, backs up every touched file to
  ~/.kernelport/backups/<ts>/, records install state in ~/.kernelport/setup_state.json. Claude Code uses
  `claude mcp add --scope user` when the CLI is on PATH (it rewrites ~/.claude.json itself), JSON edit
  otherwise. Codex TOML gets a marker-delimited block validated with tomllib. Cursor entry sets
  KERNELPORT_ROOT=${workspaceFolder}. Uninstall removes only what was added, restores replaced entries and
  skill dirs, and restores the original file byte-for-byte when nothing else changed. Server command =
  this interpreter `-m kernelport mcp` (or `uvx kernelport mcp` from an ephemeral uvx/pipx env).
  13 setup tests (fake home dir). 181 tests pass. On this machine only Claude Code is detected.
  NOT yet run against the real ~/.claude.json: that needs the owner's go-ahead (M9 step 1).

- **M10 Report** (2026-10-13): `kernelport/report/data.py` collects everything once (statuses from
  `Workspace.status()`, frozen-reference provenance, attempt history and attempts-to-first-PASS, latest
  failure report that still applies, tolerances used, benchmarks marked stale when the port changed);
  `report/html.py` renders a self-contained page (summary tiles, kernel x backend matrix, per-kernel
  details, light/dark, no network loads, all values escaped, data embedded as JSON) and `report/markdown.py`
  the same data for PR comments / agent replies. `make_report(format="html"|"markdown")` (markdown also
  returned inline as `content`); CLI `kernelport report [--md]`. SKILL.md step 9 uses the markdown report.
  6 report tests on a synthetic workspace (no GPU). 187 tests pass.

- **M11 CPU (Numba)** (2026-10-13): `backends/cpu/runner.py`: port = `run(...)` over C-contiguous NumPy
  arrays with launch.json shapes, outputs written in place (returning a value is an error), loaded once per
  file hash so Numba compiles once per dtype, wall-clock timing marked not comparable. dtypes float32/float64
  (fp16/bf16 skipped and named). `cpu_name()` now reads the real CPU model (registry / sysctl /
  /proc/cpuinfo) + thread count, e.g. "Intel(R) Core(TM) i7-14700HX (28 threads)". Hand-written Numba ports
  for all 12 zoo kernels: all PASS (float32). 6 bad cpu ports (missing tail, returns array, rebinds output,
  exclusive scan, overwrites inout, unbiased variance) FAIL for the right reason. Found while writing them:
  every reduce_sum case started out at 0, so a port that overwrote the inout accumulator would have passed;
  added case `accumulate_1000` (out starts ~10), the original CUDA kernel passes it. `skills/.../cpu.md`
  (mapping table, patterns, precision, checklist; Numba facts checked on 0.68: float literal promotes to
  float64, scalar +=/max reductions work in prange, array += in prange races and loses updates, float16
  unsupported, no import inside njit). `bind_kwargs` moved to backends/base.py. Zoo table marks
  non-comparable speedups "(n/c)". All cpu tests run without a GPU. 212 tests pass.

- **Zoo Triton ports complete** (2026-10-13): Triton ports for the remaining 8 kernels (saxpy, transpose,
  reduce_sum with tl.atomic_add, layernorm/rmsnorm with tl.sqrt_rn, rope, embedding_gather, cumsum with
  tl.cumsum + carry). All 12 Triton ports PASS every case in fp32/fp16/bf16 on the RTX 5060; speedups vs
  the original 1.00-1.30x (memory bound), matmul 6.2x (tiled tl.dot vs naive). cpu softmax stores exp once
  (64 -> 35 ms). README has the full results table.
- **M16 docs (part)** (2026-10-13): README rewritten (pipeline, quick start, backends, zoo results, CLI/MCP
  table), docs/installation.md (extras, NVIDIA / CPU-only installs, agent setup), docs/how-it-works.md
  (why each verification step exists, statuses, workspace layout). 222 tests pass.

- **M9 Dogfood** (2026-10-13, approved by owner): `kernelport setup --agent claude` registered the server via
  `claude mcp add`; `claude mcp list` showed it Connected and the skill was discovered. Instead of a separate
  `claude -p` session (would spend the owner's credits), the agent workflow was run by hand through the real
  MCP server over stdio (same command Claude Code uses) on a scratch repo with only rope/layernorm/cumsum .cu
  files, following SKILL.md. Result: 3/3 references frozen on the first validation, 6/6 ports PASS (Triton +
  cpu); 2 ports needed one repair each (layernorm: masked lanes added mean^2 to the variance; cumsum: carry
  computed as max instead of sum), both diagnosed from the failure report. Fixes from findings:
  read_kernel returns compact params (name/type/kind); common_bugs gained the "masked lanes feed a reduction"
  symptom row; suggested specs note bfloat16 for T-templated kernels. Uninstall left an empty top-level
  `"mcpServers": {}` (created by `claude mcp add`, kept by `remove`): `--uninstall` now deletes it when the
  original had none and the file round-trips byte-exactly (CRLF-safe). ~/.claude.json was then verified
  identical to the pre-setup snapshot (all top-level keys and values), skill dir removed, ~/.kernelport removed.

- **M12 WebGPU** (2026-10-13): runner on wgpu-native (`wgpu` 0.32 wheel, 3.4 MB; owner approved small
  downloads only, so Playwright Chromium was NOT used). Port = Python file with `WGSL` (entry `main`) and
  `grid(...)`. The runner appends the bindings: `alias T`, one storage buffer per tensor param named like
  the param (read for inputs, read_write for outputs/inouts), int64 as vec2<u32>, and a `params` uniform
  struct of scalars; `BINDINGS` overrides a buffer type (atomics). Only core WebGPU features + shader-f16
  are enabled, so passing WGSL is browser-portable. Explicit bind group layouts, pipelines cached per shader
  text. dtypes: float32, float16 when the adapter has shader-f16 (Backend.dtypes() now asks the capability
  module, so f16 is skipped, not failed, elsewhere). Worked around a wgpu 0.32 bug
  (queue.on_submitted_work_done_sync has a wrong C callback signature) by syncing with a 4-byte readback.
  All 12 zoo kernels have WGSL ports: all PASS fp32+fp16 on the RTX 5060 via Vulkan. 4 bad ports
  (no grid-stride on 16M, >65535 workgroups, wrong buffer name -> compile error with WGSL line numbers,
  racy non-atomic out[0] +=) FAIL for the right reason. Shader error excerpts are ASCII-fied (box chars
  crashed cp1252 consoles). Doctor smoke test for webgpu. skills/.../webgpu.md written.

## Next (pick up here)
- M13 metal (Mac), M14 AMD/HIP, M15 optional agent loop, rest of M16 docs.

## How to resume
- Env: `.venv` (Python 3.12, torch 2.11.0+cu128, triton-windows 3.7.1, cupy-cuda12x 14.2). `kernelport doctor`.
- Tests: `.venv\Scripts\python.exe -m pytest -q` (~3 min, GPU tests skip without CUDA).
- Repo: https://github.com/PS12007/kernelport (public), commit after each milestone, update this file.
- Gotcha for automated edits on this Windows box: bash heredocs mangle backslash escapes (\n, \b) inside
  Python source. Write files with the editor tool or a script file instead.

## Known issues / notes
- Requires mcp>=2.0 (FastMCP was renamed MCPServer in 2.x; we use the 2.x API only).
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
