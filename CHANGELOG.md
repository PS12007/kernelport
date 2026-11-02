# Changelog

## 0.2.0 (2026-11-02)

**Multi-kernel pipelines**
- Run ported kernels in a chain and check every stage on the data it really gets
  (`kernelport pipeline`, MCP tools `write_pipeline` and `verify_pipeline`).
- Time the chain of ports against the chain of original CUDA kernels, whole and per stage
  (`kernelport pipeline <name> --bench`, MCP tool `bench_pipeline`). A stage is called slower only if
  it loses every one of 5 rounds.
- Pipelines show up in `kernelport status` and the HTML/Markdown reports.

**New static checks** (warnings from `scan_kernels`, the hazards that pass runtime tests on today's
NVIDIA GPUs and break elsewhere)
- A lane can exit before a full-mask warp shuffle (`__shfl_*_sync(0xffffffff, ...)`). This flags the
  two softmax kernels fixed upstream in karpathy/llm.c#867.
- Some threads can miss a `__syncthreads()`: a thread-dependent early return before it, or the
  barrier inside a branch or loop that differs between threads.

**WebGPU**
- The runner refuses WGSL that wgpu-native accepts but browsers reject or run differently: a
  non-uniform `workgroupBarrier()`, float literals beyond `f32` range, and a `var` without initializer
  inside a loop (wgpu-native 29 zeroes it once instead of every iteration).
- In-browser check (`docs/web`): runs 15 WGSL ports (the 12 zoo kernels, two llm.c kernels and the
  FlashAttention forward pass) on the viewer's GPU against kernelport's own cases and tolerances,
  float32 and float16. CI runs it in headless Chrome. `docs/index.html` is a landing page for
  GitHub Pages.
- The Colab notebook has a self-test section that runs the zoo and the llm.c example on the T4.
- `shader-f16` support is checked when a shader runs, not when binding text is generated.

**Analyzer**
- Kernels defined through macros (`#define KERNEL __global__`, `FOR_TYPES(DEFINE_X)`, `C_KERNEL(name)`)
  are found, with a small cpp-style macro expander.
- Survives braces opened separately in each `#if` branch.

**Examples**
- `examples/flash-attention-minimal`: the unmodified `flash.cu` ported to Triton (23x the original),
  Metal and WGSL, all PASS.
- `examples/llm.c`: two pipelines (GPT-2 loss head; bf16 encoder, GELU, residual).

**CLI and reports**
- `kernelport verify softmax` and `bench softmax` take the kernel id as an argument; unknown ids are
  usage errors.
- Reports lead with a headline (kernels verified, PASS results, hardware targets) and each PASS cell
  shows the hardware and speedup.
- The README has an animated recording of a real session (`scripts/record_demo.py`).

**Fixes**
- Benchmarks of very short kernels no longer flag noise as a slowdown.
- WebGPU tests: a deliberately racy port may pass on a CPU adapter, so that case is skipped there
  instead of failing.

## 0.1.0 (2026-10-14)

First release: MCP server, agent skill and CLI; Triton, Metal, WebGPU and CPU backends verified,
HIP written but not run on AMD hardware; 12-kernel zoo; the llm.c example.
