# kernelport: status and launch plan

*Updated 2026-10-15. kernelport 0.1.0 on PyPI; see HANDOFF.md for the latest one-page status. Technical history: [PROGRESS.md](PROGRESS.md).*

## Where we are

| Measure | Estimate | Why |
|---|---|---|
| Public v0.1 (usable, installable, honest) | **~95%** | Released to PyPI, CI green on 3 OSes + Apple Silicon + 2 WebGPU adapters, real-code example, fresh-agent test passed. Missing: AMD verification. |
| Launch-ready (could be posted for attention) | **~80%** | Missing: a demo video/GIF, a first run on Linux + NVIDIA (only Windows + NVIDIA so far), launch posts. |
| Mature product | **~60%** | Lexical analyzer (no C++ front end), single-kernel focus, no multi-kernel pipelines, HIP unverified, only one agent (Claude) tested. |

### Backend status

| Backend | Status | Evidence |
|---|---|---|
| Triton | working | 12/12 zoo + 9/9 llm.c ports PASS on RTX 5060 (Windows); 1.0-1.3x the original CUDA on the zoo, matmul 6.2x |
| Metal (MLX) | working | 12/12 zoo + 5/5 llm.c ports PASS on Apple M1 in CI, every push |
| WebGPU (WGSL) | working | 12/12 zoo PASS on NVIDIA (Vulkan), Apple (Metal) and llvmpipe (CI); 2/2 llm.c fp32 ports PASS on all three |
| CPU (Numba) | working | 12/12 zoo PASS (fp32) |
| HIP (hipRTC) | **unverified** | ports written; `scripts/amd_check.sh` ready for an AMD machine |

### What has been proven

- **Real code:** `examples/llm.c` has 9 kernels from unmodified llm.c files (host code, cuBLAS headers,
  `Packed128` templates), 20 PASS results across 5 hardware targets.
- **A real bug found:** validation exposed that llm.c's `softmax_forward_online_kernel8` leaves rows
  unwritten when C < block size. Fix submitted upstream: karpathy/llm.c#867.
- **Fresh agent:** a coding agent with no context ported 4 new kernels; 4/4 PASS on the first
  verification. Its feedback was fixed.
- **Install:** `pip install kernelport` from PyPI, clean install checked.
- **Honesty checks:** deliberately broken ports fail for the right reason on every backend; tuning
  and benchmarks refuse to report noise as speedups.

## Next steps

### Owner actions (need an account, hardware or a decision)

1. **AMD:** get AMD Developer Cloud credits (AMD AI Developer Program), start an MI300X with the
   ROCm image, run `bash scripts/amd_check.sh`, send back `amd_results.txt`.
2. **Colab:** run `notebooks/freeze_references_colab.ipynb` once on a free T4. That also gives the
   first Linux + NVIDIA run.
3. **Demo video/GIF (30-60 s):** clone a CUDA-only repo, one prompt to the agent, the green status
   table. This is the single biggest launch lever.
4. **Launch posts, same day:** Show HN, r/LocalLLaMA, r/MachineLearning, X (MLX and Triton
   communities). Lead with "proven correct, not just translated" and the llm.c bug.
5. Optional: try Codex / Cursor / Gemini CLI with `kernelport setup` (all free tiers work; no
   API credits needed anywhere).

### Engineering that would make it better

- **Linux GPU in CI:** none of the free runners have one; a Colab or Kaggle notebook run per
  release would cover Linux + NVIDIA.
- **Multi-kernel pipelines:** verify a sequence (e.g. layernorm -> matmul -> gelu) end to end,
  not just single kernels.
- **More real projects:** e.g. a FlashAttention-style kernel or a cuBLAS-heavy repo.
- **Browser demo page:** paste a CUDA kernel, watch the WGSL port verified on your own GPU.
- **Report polish:** the HTML report is what people will share.
- **Physical Mac timings:** CI's virtual M1 verifies correctness but gives no fair speed numbers.

## Could it trend on GitHub?

It has a real chance with a deliberate launch. In its favour: a timely hook (CUDA lock-in, Apple
Silicon, coding agents), a distinctive angle (other tools translate; none prove ports correct on
real hardware), a recognisable real-code example (llm.c) plus a real upstream bug fix, and a
two-minute install. Against: without a demo video it stays abstract, and the "NVIDIA needed once
for validation" step needs explaining (the Colab notebook answers it).

## Rules kept during development

- No large downloads; anything downloaded for testing is uninstalled afterwards.
- No API credits spent (owner's choice; nothing requires them).
- Any edit to real agent config (`~/.claude.json`) is reverted after testing.
- Nothing is posted publicly (issues, PRs) without the owner's go-ahead.
