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

## Next
- M2 Complete the zoo (CUDA + launch.json + reference for the remaining 9 kernels). All originals must pass.

## Known issues / notes
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
