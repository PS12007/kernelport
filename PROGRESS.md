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

- **M15 Built-in agent (optional)** (2026-10-13): `kernelport/agent.py` + `kernelport agent "<task>"`.
  Manual tool-use loop (not the beta tool runner, so the budget/tool surface/client stay testable) over the
  11 tools; tool schemas generated from the tool signatures with `root` removed (the model cannot choose
  the workspace); system prompt = preamble + MCP instructions + SKILL.md, cached (stable, no timestamps);
  default model claude-opus-5-5, effort high, max_tokens 16000, turn budget 80; refusal / max_tokens /
  max_turns stop the loop; parallel tool results returned in one message; tool exceptions become
  is_error results; server-side refusal fallbacks (`fallbacks: "default"`) on by default. Asks for
  confirmation before spending. The SDK is imported only when a real client is built
  (`pip install kernelport[agent]`). NOT run against the live API (owner: no API credits); 7 tests with a
  scripted fake client cover the loop.

- **M13 Metal + M14 HIP** (2026-10-13; Metal verified in CI on 2026-10-14, HIP still unverified):
  - metal: `backends/metal/runner.py` on `mx.fast.metal_kernel`. Port = Metal kernel body `SOURCE`,
    `THREADGROUP`, `grid()` in TOTAL threads; inputs = input/inout tensors then scalars as 1-element
    arrays; outputs = outputs + inouts as `<name>_out`; template T; NaN `INIT_VALUE`; `ATOMIC_OUTPUTS`;
    `HEADER`; bf16 converted through float32 (exact). 12 zoo metal ports (precise:: math). Tests use a fake
    `mlx.core` to check the exact metal_kernel build/call arguments; real verification is skipped off-Mac.
  - hip: `backends/hip/runner.py` reuses the CuPy RawModule path (hipRTC on ROCm CuPy) with HIP type names
    (`__hip_bfloat16`); launched exactly like the original (launch.json grid/block). `run_cuda.get_kernel`
    now takes type names / prelude / options. 12 zoo hip ports: 11 are header swaps of the originals,
    rmsnorm rewritten with a wave-size-independent shared-memory reduction (CUDA used 32-lane shuffles;
    AMD CDNA wavefronts are 64). Launch path tested for real on NVIDIA (HIP ports compiled with NVRTC after
    mapping headers back): vector_add, rmsnorm, reduce_sum, softmax PASS all dtypes. hipRTC acceptance and
    AMD results are unverified until someone runs `kernelport zoo --backend hip` on ROCm.
  - skills/.../metal.md and hip.md, both stating their unverified status.

- **Launch prep, session 2** (2026-10-14):
  - **CI** (`.github/workflows/ci.yml`): CPU tests on ubuntu / windows / macos (CPU torch + numba), wheel
    build + `twine check` + core-only smoke install, and a **Metal job on `macos-14`**. GitHub's arm64
    macOS runners expose a real Metal device ("Apple M1 (Virtual)"), so the job verifies all 12 zoo metal
    ports for real: **12/12 PASS every case in fp32/fp16/bf16**. The job is blocking and asserts that
    Metal is available, so it cannot silently skip. `release.yml`: tag `v*` -> build -> PyPI via trusted
    publishing (needs one-time setup on pypi.org; name `kernelport` was free on PyPI and TestPyPI).
  - First Mac run showed reduce_sum "FAIL 4/16 (actual 0.0)". Not a Metal bug: the fake-MLX contract tests
    left a fake kernel in `runner._kernels`, and the real-Mac test reused it (the 4 "passing" cases
    expected 0). Fixed with an autouse fixture plus `id(mx)` in the cache key. Diagnosed on a throwaway
    branch (deleted).
  - Content hashes are line-ending insensitive (`kernelport/hashing.py`): a `.kernelport/` frozen on
    Linux/Colab and checked out on Windows with `core.autocrlf=true` used to read as "tampered".
  - `kernelport validate --all` (`Workspace.pending_validation`) and `notebooks/freeze_references_colab.ipynb`:
    write references anywhere, freeze them once on any NVIDIA GPU (free Colab T4), verify anywhere.
    SKILL.md tells agents what to do on UNAVAILABLE. Installation docs cover Mac and the no-NVIDIA flow.
  - **`tune_port`** (12th tool, `kernelport tune`): Python ports declare module-level constants and a literal
    `TUNE` dict (<= 32 combos). Every variant is verified on every case, passing ones are timed in 5
    interleaved rounds; the port is replaced only if the median gain >= 3% AND every winner round beats
    every current round, then re-verified. Without interleaving, laptop-GPU clock ramp-up made softmax look
    1.22x faster with another BLOCK; interleaved, all variants were within 2% and the port was kept.

  - **WebGPU in CI**: Linux llvmpipe (software Vulkan, 128 MiB binding limit) and macOS Metal ("Apple
    Paravirtual device"): 12/12 WGSL ports PASS on both. embedding_gather's biggest cases first failed:
    wgpu validation error on llvmpipe (500 MiB table > 128 MiB), silent all-zero output on paravirtual Metal.
    A throwaway diagnostic showed the paravirtual GPU claims 3.5 GiB per binding, but a dispatch binding
    628 MiB total does not run at all (500 MiB works). New `CaseNotRunnable`: runners raise it for
    device limits; `run_cases` names those cases in `reason` ("cases this device cannot run, not tested")
    and never counts them as passed (same rule as unsupported dtypes). The webgpu runner checks per-binding
    limits plus a 500 MiB per-dispatch cap only for the paravirtual device. CI webgpu job is blocking.

  - **Real code: `examples/llm.c`** (llm.c @ f1e2ace, MIT, files unmodified). New
    `harness/device_source.py` slices real .cu files to their device code for NVRTC (drops host
    functions/globals/prototypes and host `#include <...>`, inlines local includes once with `#line`
    markers, keeps line numbers, re-adds cuda_fp16/bf16 when a CUDA host header was dropped). All 23
    llm.c kernels compile; 5 got references + specs (4 validated first try), Triton ports PASS on the
    RTX 5060 (0.9-1.4x), Metal ports PASS first run on Apple M1 in CI against the NVIDIA-frozen refs.
    Validation found softmax_forward_online_kernel8 needs C >= blockDim.x (C=31/32/255 FAIL, 256/257 PASS).
  - Fixes from that run: bench/tune default to the spec's runnable dtype (bf16-only kernels were refused);
    the CUDA-event timer now queues launches behind `torch.cuda._sleep` (short Triton kernels were
    measured as Python launch overhead: encoder 0.36x -> 0.9x real); tune reuses one input set (the GPU
    clocked down while big cases were regenerated) and deletes variant files; `kernelport status` is
    readable when piped (160 cols, short names, idle kernels folded, `--all`).

  - **Fresh-agent dogfood** (2026-10-14): a general-purpose agent with no context got only "port the CUDA
    kernels in this repo to Triton and verify them", the kernelport CLI and its skill text, on a fresh repo
    with 4 untouched llm.c kernels (adamw x2, crossentropy forward, fused crossentropy/softmax backward).
    Result: 4/4 references validated (one after a fix), 4/4 Triton ports PASS on the first verify, 0 repair
    budget used. Its friction list drove these fixes: float scalars are rounded to their declared dtype for
    everyone (its adamw reference failed on unrounded 0.999); SKILL.md documents float dims, safe `init`
    domains, runtime block sizes, int-overflow/size limits, what `source` means; the CLI no longer crashes
    on cp1252 consoles (piped output is UTF-8) and tables use 160 cols when piped; `tune` says why it kept
    a port; `bench` runs 3 alternating rounds and prints ranges plus a >20% noise warning; doctor help names
    its MCP tool. Its 4 kernels were replayed (files unchanged) into examples/llm.c (now 9 kernels).
  - llm.c softmax bug: a fix was tested (remove the early return, guard the -inf merge: 10/10 C values
    PASS); issue text drafted for the owner, not posted.

## Next (pick up here)
Owner actions (see RELEASING.md for PyPI):
- PyPI: DONE 2026-10-14, kernelport 0.1.0 published via release.yml (tag v0.1.0); clean `pip install
  kernelport` verified. Next release: bump version, tag `vX.Y.Z`.
- llm.c softmax fix: posted by the owner as karpathy/llm.c#867 (branch fix-softmax-online-kernel8-small-c
  on PS12007/llm.c). Watch for review comments.
- HIP + Triton-on-ROCm: run `bash scripts/amd_check.sh` on an AMD machine (AMD Developer Cloud
  credits via the AMD AI Developer Program); it writes amd_results.txt. Live `kernelport agent` test: skipped
  by owner choice (no API credits); the MCP/CLI path covers the same tools;
  run notebooks/freeze_references_colab.ipynb once on Colab; dogfood with Codex / Cursor too.
Engineering left: kernels generated by macros (analyzer + slicer are lexical); a demo GIF/video;
launch posts.

## How to resume
- Env: `.venv` (Python 3.12, torch 2.11.0+cu128, triton-windows 3.7.1, cupy-cuda12x 14.2). `kernelport doctor`.
- Tests: `.venv\Scripts\python.exe -m pytest -q` (~6 min, GPU tests skip without CUDA). CI runs on
  every push to main: https://github.com/PS12007/kernelport/actions
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
