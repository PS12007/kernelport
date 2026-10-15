# kernelport: handoff notes

*Last updated 2026-10-14. Read this first when resuming. Deeper detail: [PROGRESS.md](PROGRESS.md)
(full technical history), [LAUNCH_PLAN.md](LAUNCH_PLAN.md) (launch strategy), [RELEASING.md](RELEASING.md)
(publishing).*

## Where things stand

kernelport 0.1.0 is on [PyPI](https://pypi.org/project/kernelport/), and the llm.c bug it found is
fixed upstream in an open PR, [karpathy/llm.c#867](https://github.com/karpathy/llm.c/pull/867).
CI is green on Linux, Windows, macOS, Apple Silicon (Metal) and two WebGPU adapters. HIP (AMD) is the
only backend never run on its real hardware.

| Measure | Estimate | What's missing |
|---|---|---|
| Public v0.1 (installable, works, honest about status) | ~95% | AMD verification |
| Ready to launch publicly | ~80% | Demo video, a first Linux + NVIDIA run, launch posts |
| Mature product | ~65% | Multi-kernel pipelines, more agents and real projects |

## What is verified

| Backend | Status | Evidence |
|---|---|---|
| Triton | Working | 12/12 zoo + 9/9 llm.c ports pass on RTX 5060 (Windows); 1.0-1.3x the original CUDA, matmul 6.2x |
| Metal (MLX) | Working | 12/12 zoo + 5/5 llm.c ports pass on Apple M1, checked by CI on every push |
| WebGPU (WGSL) | Working | 12/12 zoo pass on NVIDIA (Vulkan), Apple (Metal) and llvmpipe; 2/2 llm.c fp32 ports pass on all three |
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
| Failure report | "unexpected NaN" hint also names overflow (inf/inf), not only unwritten elements |
| CI | Flaky WebGPU test fixed: the racy reduce_sum bad port can PASS on llvmpipe (workgroups run one by one), so it is skipped there; earlier red runs on 2026-10-14 were this flake |

## Next steps (owner)

None of these need API credits.

- [ ] **AMD:** join AMD's AI Developer Program for free Developer Cloud credits, start an MI300X with the
      ROCm image, run `bash scripts/amd_check.sh`, send back `amd_results.txt`.
- [ ] **Colab:** run `notebooks/freeze_references_colab.ipynb` once on a free T4 (also the first Linux +
      NVIDIA run).
- [ ] **Demo video/GIF (30-60 s):** clone a CUDA-only repo, one prompt to the agent, the green status
      table. The biggest launch lever. (The README now has an animated CLI recording; a video of the
      agent itself doing it is still the stronger asset.)
- [ ] **Browser check online:** GitHub repo Settings → Pages → deploy from `main`, folder `/docs`. The
      page is then at https://ps12007.github.io/kernelport/web/ ("check the ports on your own GPU",
      a good launch link). Nothing else is needed; it is static.
- [ ] **Launch posts, same day:** Show HN, r/LocalLLaMA, r/MachineLearning, X; lead with "proven
      correct, not just translated" and the llm.c bug.
- [ ] **PR #867:** watch for review comments (llm.c is quiet; replies may be slow).
- [ ] Optional: try Codex, Cursor or Gemini CLI with `kernelport setup`.

Future releases: bump the version in `pyproject.toml` and `kernelport/__init__.py`, then push a
`vX.Y.Z` tag (see RELEASING.md).

## Ideas to make it better

- **Multi-kernel pipelines:** verify a chain end to end (e.g. layernorm -> matmul -> gelu), not just single kernels.
- **More real projects:** a FlashAttention-style kernel or a cuBLAS-heavy repo.
- **Linux GPU checks:** free CI runners have no GPU; a Colab or Kaggle run per release would cover Linux + NVIDIA.
- **Browser demo page:** paste a CUDA kernel and watch the WebGPU port verified on your own GPU.
- **Report polish:** the HTML report is what people will share.
- **Real Mac timings:** CI's virtual M1 proves correctness but gives no fair speed numbers.

## How to resume

Open Claude Code in `C:\-CS\CUDAtoanything` and say "keep working on kernelport".

| What | Where |
|---|---|
| Repo | https://github.com/PS12007/kernelport |
| Package | https://pypi.org/project/kernelport/ |
| CI runs | https://github.com/PS12007/kernelport/actions |
| llm.c PR | https://github.com/karpathy/llm.c/pull/867 (branch `fix-softmax-online-kernel8-small-c` on PS12007/llm.c) |
| Environment | `.venv` (Python 3.12, torch 2.11 cu128, triton-windows 3.7, cupy 14.2); run tools as `.venv\Scripts\python.exe` / `.venv\Scripts\kernelport.exe` |
| Tests | `.venv\Scripts\python.exe -m pytest -q` (~6 min; 271 pass, 43 skip for hardware not on this machine) |

## Rules to keep

- No large downloads; test installs (e.g. `wgpu`) are removed afterwards.
- No API credits, ever (owner's choice; nothing requires them).
- Edits to the real agent config (`~/.claude.json`) are reverted after testing.
- Nothing is posted publicly (issues, PRs, tags, releases) without the owner's go-ahead.
- On this Windows box, write code with the editor tools, not shell heredocs (they mangle backslashes and backticks).
