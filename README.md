# kernelport

[![CI](https://github.com/PS12007/kernelport/actions/workflows/ci.yml/badge.svg)](https://github.com/PS12007/kernelport/actions/workflows/ci.yml)

Bring CUDA-only kernels to Mac, AMD, browsers, and CPU, with proof they are correct.

kernelport is an **MCP server + agent skill + CLI**. Your coding agent (Claude Code, Codex,
Cursor, Gemini CLI) writes the ports; kernelport runs them on real hardware and checks every
one against a reference that was itself validated against the original CUDA kernel. A port is
only ever reported as **PASS** if its current code ran on real hardware and matched.

> Status: early development (0.1.0.dev0). Triton, CPU and WebGPU backends work and are verified;
> Metal is verified on Apple Silicon in CI; the HIP runner and zoo ports are written but not yet
> verified on an AMD GPU. See [PROGRESS.md](PROGRESS.md).

## How it works

```
your repo: kernels/*.cu
      │  scan_kernels                    (lexical CUDA analyzer: params, kinds, features)
      ▼
 PyTorch reference + input spec          written by the agent
      │  validate_reference              runs the ORIGINAL CUDA kernel on every case
      ▼
 frozen reference (sha256 manifest)      only a passing validation can freeze it
      │  write_port / verify_port        port vs frozen reference, every case and dtype
      ▼
 PASS / FAIL (+ compact failure report) / UNVERIFIED (backend not available here)
      │  bench_port                      only after PASS on this machine
      ▼
 report.html / report.md
```

The agent never sets tolerances, never edits the frozen reference, and gets at most 6 failed
verification attempts per port. Those rules are enforced in the tools, not just in the
skill. Details: [docs/how-it-works.md](docs/how-it-works.md).

## Quick start

```bash
# 1. install (CUDA build of torch first; see docs/installation.md)
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install "kernelport[cuda,triton,cpu] @ git+https://github.com/PS12007/kernelport"

# 2. check what can run here
kernelport doctor

# 3. register the MCP server and skill with your agents (shows every change, asks first)
kernelport setup            # --print to only show instructions, --uninstall to undo
```

Then, in your repo, ask your agent:

> port the CUDA kernels in this repo to Triton and verify them

## Backends

| backend | port language | runs on | status |
|---|---|---|---|---|
| `triton` | Triton (Python) | NVIDIA (triton / triton-windows), AMD ROCm | working |
| `cpu` | Numba (Python) | any machine | working (float32/float64) |
| `metal` | Metal kernel bodies via MLX `metal_kernel` | Apple Silicon | working (float32/float16/bfloat16), verified in CI on an Apple M1 |
| `webgpu` | WGSL | any Vulkan / Metal / D3D12 GPU via wgpu-native (core WebGPU only, so browser-portable) | working (float32/float16); verified on NVIDIA (Vulkan), Apple (Metal) and llvmpipe (software Vulkan) |
| `hip` | HIP C++ via hipRTC (CuPy ROCm) | AMD ROCm | written, launch path tested on NVIDIA, **not yet verified on AMD** |

Ports for a backend that is not available on the current machine are saved as **UNVERIFIED**,
never PASS. Commit `.kernelport/` and run `kernelport verify` on the right machine to verify them there.

## No coding agent? Use the built-in one (optional)

```bash
pip install "kernelport[agent] @ git+https://github.com/PS12007/kernelport"
export ANTHROPIC_API_KEY=...        # or: ant auth login
kernelport agent "port the CUDA kernels in this repo to Triton and verify them"
```

`kernelport agent` runs Claude (default `claude-opus-5-5`) in a tool-use loop over the same 12
tools the MCP server exposes, with the workflow skill as its system prompt. It can only act
through those tools, so all writes stay in `.kernelport/` and it cannot run shell commands.
Runs are billed to **your** Anthropic account, so it asks before starting (`--yes` skips the
prompt). The run stops at `--max-turns` (default 80). Server-side refusal fallbacks are on by
default (`--no-fallbacks` to disable).

## The kernel zoo

`zoo/` holds 12 CUDA kernels with launch specs and references: vector_add, relu, saxpy,
transpose, reduce_sum, softmax, layernorm, rmsnorm, matmul_naive, rope, embedding_gather and
cumsum. Between them they use templates, shared memory, `__syncthreads`, warp shuffles,
atomics, block reductions, scans, gathers and int64 indices. Every original passes its reference.
`kernelport zoo --backend triton|cpu` validates them and verifies the stored ports.

Measured on an RTX 5060 Laptop GPU (sm_120) and an i7-14700HX (28 threads), bench case of each spec
(the largest), float32. Every stored port passes every case: Triton in fp32/fp16/bf16, cpu in fp32, webgpu in fp32/fp16, metal in fp32/fp16/bf16 (on an Apple M1 GitHub Actions runner, checked on every push; no timings, a virtualized runner is not a fair benchmark). CI also runs the WGSL ports on Apple Metal and on llvmpipe; there the 628 MiB embedding_gather case exceeds the adapter binding limit and is reported as not tested.

| kernel | triton vs original CUDA | cpu (Numba) | webgpu (WGSL) | metal (MLX) |
|---|---|---|---|---|
| vector_add | PASS, 1.00x | PASS, 7.8 ms (n/c) | PASS, 1.12 ms (n/c) | PASS |
| relu | PASS, 1.04x | PASS, 5.7 ms (n/c) | PASS, 1.06 ms (n/c) | PASS |
| saxpy | PASS, 1.00x | PASS, 6.4 ms (n/c) | PASS, 1.08 ms (n/c) | PASS |
| transpose | PASS, 1.00x | PASS, 11.2 ms (n/c) | PASS, 1.06 ms (n/c) | PASS |
| reduce_sum | PASS, 1.04x | PASS, 3.4 ms (n/c) | PASS, 0.58 ms (n/c) | PASS |
| softmax | PASS, 1.00x | PASS, 35.5 ms (n/c) | PASS, 0.67 ms (n/c) | PASS |
| layernorm | PASS, 1.00x | PASS, 9.1 ms (n/c) | PASS, 0.96 ms (n/c) | PASS |
| rmsnorm | PASS, 1.00x | PASS, 7.0 ms (n/c) | PASS, 1.13 ms (n/c) | PASS |
| matmul_naive | PASS, 6.20x (tiled `tl.dot`, IEEE fp32) | PASS, 40.3 ms (n/c) | PASS, 4.21 ms (n/c) | PASS |
| rope | PASS, 1.00x | PASS, 13.1 ms (n/c) | PASS, 1.42 ms (n/c) | PASS |
| embedding_gather | PASS, 1.02x | PASS, 11.8 ms (n/c) | PASS, 1.46 ms (n/c) | PASS |
| cumsum | PASS, 1.30x | PASS, 6.9 ms (n/c) | PASS, 1.18 ms (n/c) | PASS |

(n/c = wall-clock time (CPU, or WebGPU including submission), not comparable to GPU kernel time.)

## CLI

Every MCP tool has a CLI equivalent, so you can drive the workflow by hand or from CI:

| command | MCP tool | what it does |
|---|---|---|
| `kernelport doctor` | `kernelport_doctor` | OS, GPU, extras, each backend with a reason |
| `kernelport scan [path]` | `scan_kernels` | find `__global__` kernels (also CUDA strings in .py) |
| `kernelport read <id>` | `read_kernel` | source, parsed signature, suggested input spec |
| `kernelport skill [name]` | `read_skill` | workflow, common_bugs, or a backend guide |
| `kernelport write-reference` | `write_reference` | save a candidate reference + spec |
| `kernelport validate <id>` or `--all` | `validate_reference` | original CUDA vs reference; freeze on PASS |
| `kernelport write-port` | `write_port` | save a port |
| `kernelport verify` | `verify_port` | port vs frozen reference |
| `kernelport bench` | `bench_port` | time port vs original (after PASS) |
| `kernelport tune <id> <backend>` | `tune_port` | verify + time every `TUNE` variant, keep the fastest that passes |
| `kernelport status` | `list_status` | kernel × backend × hardware table |
| `kernelport report [--md]` | `make_report` | HTML or Markdown report |
| `kernelport setup` | – | register with Claude Code / Codex / Cursor / Gemini CLI |
| `kernelport mcp` | – | run the MCP server on stdio |
| `kernelport agent "<task>"` | – | optional built-in agent (your Anthropic API key) |

## Development

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -e ".[cuda,triton,cpu,dev]"
python -m pytest -q        # ~4 min; GPU tests skip without CUDA, cpu tests need only numba
```

## License

MIT
