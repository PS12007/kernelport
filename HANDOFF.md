# kernelport: handoff notes

*Last updated 2026-11-04. Read this first when resuming. Deeper detail: [PROGRESS.md](PROGRESS.md)
(full technical history), [LAUNCH_PLAN.md](LAUNCH_PLAN.md) (launch strategy), [RELEASING.md](RELEASING.md)
(publishing).*

## Where things stand

kernelport 0.2.0 is on [PyPI](https://pypi.org/project/kernelport/), and the llm.c bug it found is
fixed upstream in an open PR, [karpathy/llm.c#867](https://github.com/karpathy/llm.c/pull/867).
CI is green on Linux, Windows, macOS, Apple Silicon (Metal), two WebGPU adapters and headless Chrome
(the in-browser check). HIP (AMD) is the only backend never run on its real hardware.

| Measure | Estimate | What's missing |
|---|---|---|
| Public v0.1 (installable, works, honest about status) | ~95% | AMD verification |
| Ready to launch publicly | ~85% | Agent demo video, GitHub Pages switched on, a first Linux + NVIDIA run, launch posts |
| Mature product | ~75% | More agents tested, AMD, pipeline timing |

## What is verified

| Backend | Status | Evidence |
|---|---|---|
| Triton | Working | 12/12 zoo + 9/9 llm.c ports pass on RTX 5060 (Windows); 1.0-1.3x the original CUDA, matmul 6.2x |
| Metal (MLX) | Working | 12/12 zoo + 5/5 llm.c + flash-attention ports pass on Apple M1, checked by CI on every push |
| WebGPU (WGSL) | Working | 12/12 zoo pass on NVIDIA (Vulkan), Apple (Metal) and llvmpipe; 2/2 llm.c fp32 ports + flash-attention pass on all three; in browsers 24/24 (fp32 + fp16) on RTX 5060 + Intel UHD (Edge), and in Chrome on SwiftShader + Apple (CI) |
| CPU (Numba) | Working | 12/12 zoo pass (fp32) |
| HIP (AMD) | Unverified | Ports written; `scripts/amd_check.sh` ready to run on an AMD machine |

- **Real code:** `examples/llm.c` has 9 kernels from unmodified llm.c files (host code, cuBLAS headers,
  `Packed128` templates), 20 PASS results across 5 hardware targets.
- **Fresh-agent test:** a coding agent with no context ported 4 new llm.c kernels; all 4 passed on the
  first verification, and its feedback was fixed.
- **Install:** a clean `pip install kernelport` from PyPI was checked.

## What was done on 2026-10-14

| Area | Change |
|---|---|
| Release | 0.1.0 published to PyPI via tag-triggered `release.yml` with trusted publishing; install docs use `pip install kernelport`; `main` is 0.1.1.dev0 |
| CI | Tests on Linux, Windows, macOS; Metal zoo + llm.c on GitHub's Apple M1 runner; WebGPU on llvmpipe and Apple Metal; wheel build check |
| Real code | Host-code slicer (`kernelport/harness/device_source.py`) compiles real .cu files (main(), cuBLAS, stdio) with NVRTC; `examples/llm.c` with 9 kernels |
| llm.c bug | `softmax_forward_online_kernel8` leaves rows unwritten when C < block size; fix tested 90/90, posted by the owner as PR #867 from fork PS12007/llm.c |
| Fresh-agent test | 4/4 first-try passes; fixes from its feedback: float scalars rounded to their dtype, guide gaps, Windows console crash, clearer tune/bench output |
| Tuning | New `tune_port` tool (12th tool): verifies every variant, keeps only clear, noise-resistant wins |
| Benchmarks | Timer no longer measures Python launch overhead; 3 alternating runs with ranges and a noise warning |
| No NVIDIA locally | `kernelport validate --all` + Colab notebook to freeze references once on a free GPU |
| Device limits | Cases too big for an adapter are reported as not tested instead of failing the port |
| Robustness | Line-ending-insensitive hashes; LF workspace files; readable status table when piped |

## What was done on 2026-10-15

| Area | Change |
|---|---|
| Macro-defined kernels | `kernelport/analyzer/macros.py`, a small cpp-style expander (`#`, `##`, variadics, `FOR_TYPES(DEFINE_X)`). The analyzer finds kernels stamped out by macros, marked by `#define KERNEL __global__`, or opened by `C_KERNEL(name)(...)`, and records `generated_by`. The NVRTC slicer treats top-level macro calls as items of their own (kept if device code, blanked if host only). Tests: `tests/test_macros.py`, including a GPU validation of a macro-generated kernel |
| README demo | `scripts/record_demo.py` records a real CLI session (scan, validate, a broken softmax port FAILs, the fix PASSes, bench, status) and renders `docs/img/demo.svg`, an animated terminal, now at the top of the README. `record` needs CUDA + Triton; `render` re-draws from `docs/demo/transcript.json` |
| CLI | `kernelport verify softmax` / `bench softmax` take the kernel id as an argument; unknown ids are usage errors; piped output honours `$COLUMNS`; "found 1 kernel" |
| Report | Headline sentence, readable dark-mode links, PASS cells show hardware + speedup, unstarted kernels folded away; Markdown matches |
| Browser check | `docs/web/index.html` runs the 12 WGSL ports in the browser's WebGPU against kernelport's own cases (`scripts/build_web_demo.py`, 57 float32 cases, 2.1 MB); 12/12 PASS in Edge on the RTX 5060 and the Intel UHD iGPU; a sabotaged port FAILs. `scripts/check_web_demo.py` runs it headless |
| Browser bug found | zoo softmax WGSL used `-3.4028235e38` (above FLT_MAX): naga accepts, Chrome/Edge reject. Port and the webgpu skill guide fixed; the runner now refuses out-of-range float literals |
| FlashAttention example | `examples/flash-attention-minimal`: unmodified `flash.cu` (Apache-2.0), reference validated 6/6 within N <= 384 (beyond that the original's own `__expf`/fp32 rounding exceeds the fixed tolerance; documented, not loosened). Triton PASS at 23x, WGSL PASS on RTX 5060 / Apple / llvmpipe, Metal PASS on M1 (CI results committed). New init `{"dist": "full", "value": "-inf"}` |
| wgpu-native bug found | A `var` declared without initializer in a loop is zeroed once, not per iteration (WGSL spec, browsers), on every backend. Already fixed upstream in naga (gfx-rs/wgpu#9592, v30), but wgpu-native has no v30 release yet and wgpu-py 0.32 bundles 29.0.1.1. The runner refuses such code. Issue drafted (not posted) in `drafts/` (gitignored) |
| Browser CI | `scripts/check_web_demo.py` runs in CI's WebGPU jobs: Chrome on SwiftShader (Linux) and Apple GPU (macOS), 12/12 |
| Failure report | "unexpected NaN" hint also names overflow (inf/inf), not only unwritten elements |
| CI | Flaky WebGPU test fixed: the racy reduce_sum bad port can PASS on llvmpipe (workgroups run one by one), so it is skipped there; earlier red runs on 2026-10-14 were this flake |

## What was done on 2026-10-17

| Area | Change |
|---|---|
| Pipelines | Kernels run in a chain over named buffers (`.kernelport/pipelines/<name>.json`); every stage's port is checked against its frozen reference on the data the stage before really wrote. End-to-end drift vs a float64 chain is reported, never decides. Tools `write_pipeline` / `verify_pipeline` (14 tools), CLI `kernelport pipeline`, rows in `status` and both reports (stale results shown as UNVERIFIED). Backend `cuda` chains the originals |
| llm.c pipelines | `gpt2_loss_head` (softmax -> crossentropy -> backward) and `gpt2_embed_gelu_residual` (bf16): PASS on Triton and the originals; CI runs the bf16 one on Metal |
| Why it matters | A crossentropy port that clamps probs to >= 1e-3 passes its own cases and fails the pipeline (test). Drift shows fp32 softmax underflow (inf losses) and bf16 error growth (1.5x a tolerance after 3 kernels, same in the originals) |

## What was done on 2026-10-24

| Area | Change |
|---|---|
| llm.c PR #867 | A reviewer found the same early-return bug in `softmax_forward_kernel7`. Fixed on the PR branch (only whole warps exit; same speed), tested 504/504 configs on the RTX 5060. The unfixed kernel also passes there: undefined behaviour that happens to work, so only a static check can catch it. The PR's first commit had turned the file into CRLF (all 733 lines in the diff); restored to LF. **Committed in `C:\-CS\llm.c-fork`, not pushed** |
| Static warnings | `kernelport/analyzer/hazards.py`: a lane can return early before a full-mask warp shuffle/reduce. Shown by `scan` and `read_kernel`; the skill tells agents to test warp-splitting sizes. Flags exactly kernel7 + kernel8 in llm.c; 1 of 152 llama.cpp kernels |
| Analyzer crash | A kernel opening a brace in each `#if` branch (llama.cpp `mmf.cuh`) crashed the whole scan; fixed |

## What was done on 2026-10-27

| Area | Change |
|---|---|
| Pipeline timing | New `bench_pipeline` tool (15 tools) and `kernelport pipeline <name> --bench`: after a PASS, the chain is run once to capture each stage's real input, then the whole chain of ports and each stage are timed against the original CUDA kernels on those inputs (3 alternating rounds). Saved under `.kernelport/pipelines/results/<name>/<backend>/bench/` |
| First numbers (RTX 5060, Triton) | `gpt2_loss_head` at B=4, T=64, V=50257: chain 1.00 ms vs 1.33 ms original (1.33x; softmax 1.68x, the two cross-entropy kernels ~0.97x). `gpt2_embed_gelu_residual` (bf16, 4x256x768, ~10 µs kernels): about even (1.0-1.4x between runs) |
| Timing noise | A first run flagged a "slow" stage that moved between stages on reruns: single rounds of ~10 µs kernels spike up to 3x on this laptop. `bench_pipeline` now uses 5 rounds and names a stage slower only if every port round lost to every original round (as `tune_port` does) |
| Encoder port | `encoder_forward_kernel3` Triton port rewritten row-wise (one program per token and block of C: one token-id load, no per-element divide): 0.023 vs 0.025 ms for the old port, steady across runs; re-verified 4/4 and the bf16 pipeline PASS |
| PR #867 reply | Reply to the kernel7 reviewer drafted in `drafts/llm-c-867-kernel7-reply.md` (local, gitignored), with the push steps. Still not pushed or posted |

## What was done on 2026-10-28

| Area | Change |
|---|---|
| Barrier warnings | `analyzer/hazards.py` also warns when some threads can miss a `__syncthreads()` (or `block.sync()`, or a `blockReduce` helper): a thread-dependent early return before it, or the barrier inside a branch or loop that differs between threads. Over 340 kernels: 4 loop cases in llm.c (threads can hit the barrier a different number of times), 3 + 4 deliberate early returns in llm.c / llama.cpp (fine on NVIDIA, worded as a porting note), 0 in the zoo. Two llama.cpp false-positive patterns fixed and tested |
| WGSL gap found | wgpu-native (naga, wgpu 0.32) accepts a `workgroupBarrier()` that browsers reject as non-uniform (checked in Edge). A port could PASS in kernelport and not compile in Chrome. The runner now refuses it with a fix hint; it agrees with Edge on 6 probe shaders and accepts all 15 existing WGSL ports |
| Skill | SKILL.md, webgpu.md and metal.md: keep every thread alive to each barrier in ports, guard the work instead |

## What was done on 2026-11-04

| Area | Change |
|---|---|
| Browser check | Each kernel's WGSL port is an editor with "Run this version": the edit runs against the same cases, expected outputs and tolerances (fp32/fp16 tabs). Wrong edits FAIL with the first differing element; non-compiling ones show the compiler error. `check_web_demo.py` drives the editor too (4 checks). 27/27 + editor checks pass in Edge on the RTX 5060. Fixed the code panels: a long CUDA line no longer squeezes the WGSL side |

## Next steps (owner)

None of these need API credits.

- [ ] **AMD:** join AMD's AI Developer Program for free Developer Cloud credits, start an MI300X with the
      ROCm image, run `bash scripts/amd_check.sh`, send back `amd_results.txt`.
- [ ] **Colab:** open `notebooks/freeze_references_colab.ipynb` in Colab (T4 runtime) and run the
      cells under "Self-test" only (skip the "Your repo" cells). It runs the zoo originals + Triton
      ports and the llm.c example, the first Linux + NVIDIA run, and downloads `selftest_results.txt`.
- [ ] **Demo video/GIF (30-60 s):** one prompt to the agent, FAIL → fix → PASS, the green status
      table. The biggest launch lever. Step-by-step recording plan in `drafts/demo-video-plan.md`.
- [x] **Browser check online:** GitHub Pages is on (main, /docs) since 2026-11-02:
      https://ps12007.github.io/kernelport/ (landing) and https://ps12007.github.io/kernelport/web/.
- [ ] **Launch posts, same day:** Show HN, r/LocalLLaMA, r/MachineLearning, X. Drafts with a
      pre-post checklist in `drafts/launch-posts.md` (local, gitignored).
- [x] **PR #867:** kernel7 fix + line-ending restore pushed and the reply posted (2026-11-02); the PR
      diff is now 1 file, +8 / -8. Keep watching for review comments.
- [ ] **wgpu loop-var bug (optional):** drafts are ready, not posted, in `drafts/` (local only,
      gitignored): an issue for pygfx/wgpu-py (`wgpu-loop-var-issue.md`, with
      `wgpu-loop-var-repro.py`) and a short comment for gfx-rs/wgpu-native PR #630. Already fixed in
      naga v30 (gfx-rs/wgpu#9592); wgpu-native has no v30 release yet (re-checked 2026-11-02: latest is
      still v29.0.1.1, #630 still open). Re-check both before posting.
- [x] **Release 0.2.0:** on PyPI since 2026-11-02 (tag `v0.2.0`); `main` is 0.2.1.dev0.
- [ ] Optional: try Codex, Cursor or Gemini CLI with `kernelport setup`.

Future releases: bump the version in `pyproject.toml` and `kernelport/__init__.py`, then push a
`vX.Y.Z` tag (see RELEASING.md).

## Ideas to make it better

- **Pipelines, more:** a pipeline in the browser check; a chain with a matmul; why the original encoder's time doubles between sessions.
- **More real projects:** a cuBLAS-heavy repo, or flash-attention's backward pass (forward is done).
- **Linux GPU checks:** free CI runners have no GPU; a Colab or Kaggle run per release would cover Linux + NVIDIA.
- **Browser check, more:** larger cases as an optional download; a pipeline in the browser; share an edited port by link.
- **Real Mac timings:** CI's virtual M1 proves correctness but gives no fair speed numbers.
- **When wgpu-native ships v30:** the loop-var check can stay (harmless); note the fixed version in the message.

Done 2026-10-15 from this list: macro-defined kernels, report polish, browser demo page, FlashAttention example.
Done 2026-10-17: multi-kernel pipelines. Done 2026-10-27: pipeline timing.
Done 2026-11-02: llm.c + FlashAttention WGSL ports in the browser check. Done 2026-11-04: editable ports in the browser check.

**Handy here:** headless Edge runs WebGPU on this laptop (`scripts/check_web_demo.py`; `--discrete` for
the RTX 5060) and takes screenshots for checking pages (`msedge --headless=new --screenshot=...`).
`scripts/record_demo.py` re-records the README terminal demo.

## How to resume

Open Claude Code in `C:\-CS\CUDAtoanything` and say "keep working on kernelport".

| What | Where |
|---|---|
| Repo | https://github.com/PS12007/kernelport |
| Package | https://pypi.org/project/kernelport/ |
| CI runs | https://github.com/PS12007/kernelport/actions |
| llm.c PR | https://github.com/karpathy/llm.c/pull/867 (branch `fix-softmax-online-kernel8-small-c` on PS12007/llm.c; local clone `C:\-CS\llm.c-fork`, `core.autocrlf=false`) |
| Environment | `.venv` (Python 3.12, torch 2.11 cu128, triton-windows 3.7, cupy 14.2); run tools as `.venv\Scripts\python.exe` / `.venv\Scripts\kernelport.exe` |
| Tests | `.venv\Scripts\python.exe -m pytest -q` (~6-8 min; 332 pass, 43 skip for hardware not on this machine; with `wgpu` installed the WebGPU tests run too) |

## Rules to keep

- No large downloads; test installs (e.g. `wgpu`) are removed afterwards.
- No API credits, ever (owner's choice; nothing requires them).
- Edits to the real agent config (`~/.claude.json`) are reverted after testing.
- Nothing is posted publicly (issues, PRs, tags, releases) without the owner's go-ahead.
- On this Windows box, write code with the editor tools, not shell heredocs (they mangle backslashes and backticks).
