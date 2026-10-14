# kernelport: status and launch plan

*Written 2026-10-13, at commit `a376f38`.*

## Where we are

**About 75–80% of the way to a solid public v0.1, and roughly 50–60% of a mature product.**
All 16 planned milestones (M0–M16) are built, committed and pushed. The test suite has
261 passing and 25 skipped (the skips need a Mac, an AMD GPU, or the optional `wgpu` package).
What's left is mostly real-world proof and polish, not features.

### Backend status

| Backend | Status | Zoo ports passing |
|---|---|---|
| Triton | working, verified on an RTX 5060 Laptop GPU | 12/12 (fp32, fp16, bf16); 1.0–1.3× the original CUDA, matmul 6.2× |
| CPU (Numba) | working, verified | 12/12 (fp32) |
| WebGPU (WGSL via wgpu-native) | working, verified on the RTX 5060 through Vulkan | 12/12 (fp32, fp16) |
| Metal (MLX) | runner and ports written, **not verified on a Mac** | 12 written |
| HIP (hipRTC) | runner and ports written, launch path tested on NVIDIA, **not verified on AMD** | 12 written |

### What has been tested

- Full workflow through the real MCP server (dogfood): 3/3 references frozen on the first try,
  6/6 ports PASS, 2 needed one repair each, both diagnosed from the failure report.
- `kernelport setup` against a real Claude Code install (server connected, skill discovered),
  then `--uninstall`; `~/.claude.json` was confirmed identical to the pre-setup snapshot.
- Deliberately broken ports for every working backend fail for the right reason.
- The built-in agent (`kernelport agent`) is tested only with a scripted fake client.
  No live API calls have been made.

## Must fix before launch

1. **Verify Metal on an Apple Silicon Mac.** "Run CUDA kernels on your Mac" is the headline
   promise, and the Metal backend was written without a Mac. Run
   `kernelport zoo --backend metal` and fix what fails. Any M-series Mac, about one afternoon.
2. **Fresh-agent dogfood.** So far a person (who already knew the kernels) drove the tools.
   Give a clean Claude Code or Codex session only "port the CUDA kernels in this repo to
   Triton and verify them" and record pass rate, attempts, and where the skill was unclear.
3. **Easy install.**
   - Publish to PyPI (check first that the name `kernelport` is free).
   - GitHub Actions CI: CPU-only tests on Linux, macOS and Windows.
   - A one-line install path. Today the CUDA build of torch has to come from the PyTorch index first.
   - Only Windows has been tested so far.
4. **Make the NVIDIA reference step easy to work around.** Validating a reference runs the
   original CUDA kernel, so it needs an NVIDIA GPU, and the main target audience (Mac users)
   often has none. Options:
   - document "validate once on a GPU machine or free Colab, commit `.kernelport/`, verify anywhere";
   - ship a Colab notebook that validates and freezes references for a repo.
5. **Prove it on real code.** The zoo is 12 single-file kernels. Real CUDA projects are
   multi-file, use macros and templates heavily, and call libraries like cuBLAS. The analyzer
   works on text, not a real C++ parser, so it will miss some of this. Port at least one
   well-known CUDA-only project end to end.

## Would make it noticeably better

- **Autotuning after PASS:** try block sizes and warp counts, keep the fastest version that
  still passes. "Verified and fast" is a stronger pitch than "verified".
- **Browser demo page:** paste a CUDA kernel and watch the WGSL port get verified on your own GPU.
- **A report worth screenshotting:** the HTML report is what people will share.
- **A HIP run on AMD**, for example on a rented cloud GPU.
- **Browser check for WGSL ports** (Playwright and Chromium, about 150 MB) to confirm they
  also pass in Chrome, not just wgpu-native.
- **Live smoke test of `kernelport agent`** (needs an API key; costs credits).

## Could it trend on GitHub?

**It has a real chance, but only with a deliberate launch.** Nothing reaches trending just by
being pushed.

**In its favour**
- The hook is timely: CUDA lock-in, Apple Silicon and AI coding agents are all hot topics.
- A distinctive angle. Existing CUDA-portability tools (HIPIFY, ZLUDA, SCALE, chipStar) are
  translators or compatibility layers. As far as I know, none combine agents writing the ports
  with a harness that refuses to say PASS without proof on real hardware.
  Pitch: **"proven correct, not just translated."**
- Agent-native: one-command `kernelport setup` for Claude Code, Codex, Cursor and Gemini CLI
  makes "try it in 2 minutes" believable.

**Against it**
- Without a demo it's an abstract developer tool.
- The NVIDIA-for-validation requirement can confuse the target audience.
- Some people are sceptical of "AI ports your code". The verification angle answers that, so lead with it.

### What gets repos onto trending

1. **A 30–60 s demo video or GIF:** clone a CUDA-only repo, one prompt, a green report with
   kernels passing on a MacBook.
2. **A recognisable target:** "we ran X's CUDA kernels on a Mac, verified" beats "12 zoo kernels".
3. **Launch posts on the same day:** Show HN, r/LocalLLaMA, r/MachineLearning, and X
   (tag the MLX and Triton communities). Trending is driven by stars per day, so concentrated
   attention beats a slow trickle.
4. **A README that sells in 10 seconds:** demo GIF at the top, three-line quick start, the
   results table. The content exists; it needs the visual.

## Suggested order

1. CI + PyPI packaging (after the name check)
2. Fresh-agent dogfood run
3. Mac verification run (borrow or rent an Apple Silicon Mac)
4. Port one well-known CUDA-only project end to end
5. Demo GIF and README polish
6. Launch (same-day posts)

## Rules kept during development

- No large downloads; anything downloaded for testing is uninstalled afterwards (`wgpu` was,
  so WebGPU tests skip until `pip install kernelport[webgpu]`).
- No API credits spent.
- Any edit to real agent config (`~/.claude.json`) is reverted after testing.

Full technical history and resume notes: [PROGRESS.md](PROGRESS.md).
