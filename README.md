# kernelport

Bring CUDA-only kernels to Mac, AMD, browsers, and CPU, with proof they are correct.

kernelport is an **MCP server + agent skill + CLI**. Your coding agent (Claude Code, Codex,
Cursor, Gemini CLI) writes the ports; kernelport runs them on real hardware and checks every
one against a reference that was itself validated against the original CUDA kernel. A port is
only ever reported as **PASS** if its current code ran on real hardware and matched.

> Status: early development (0.1.0.dev0). Triton and CPU backends work; Metal, WebGPU and HIP
> are registered but not implemented yet. See [PROGRESS.md](PROGRESS.md).

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
|---|---|---|---|
| `triton` | Triton (Python) | NVIDIA (triton / triton-windows), AMD ROCm | working |
| `cpu` | Numba (Python) | any machine | working (float32/float64) |
| `metal` | MLX / Metal | Apple Silicon | planned |
| `webgpu` | WGSL | browsers (via headless Chromium) | planned |
| `hip` | HIP C++ | AMD ROCm | planned |

Ports for a backend that is not available on the current machine are saved as **UNVERIFIED**,
never PASS. Commit `.kernelport/` and run `kernelport verify` on the right machine to verify them there.

## The kernel zoo

`zoo/` holds 12 CUDA kernels with launch specs and references: vector_add, relu, saxpy,
transpose, reduce_sum, softmax, layernorm, rmsnorm, matmul_naive, rope, embedding_gather and
cumsum. Between them they use templates, shared memory, `__syncthreads`, warp shuffles,
atomics, block reductions, scans, gathers and int64 indices. Every original passes its reference.
`kernelport zoo --backend triton|cpu` validates them and verifies the stored ports.

Measured on an RTX 5060 Laptop GPU (sm_120) and an i7-14700HX, float32, bench case of each spec:

| kernel | triton | cpu |
|---|---|---|
| vector_add | PASS, 1.00x vs CUDA | PASS |
| relu | PASS, 1.02x | PASS, 5.8 ms (n/c) |
| softmax | PASS, 1.00x | PASS, 61 ms (n/c) |
| matmul_naive | PASS, 6.07x (tiled `tl.dot`, IEEE fp32) | PASS, 40 ms (n/c) |
| other 8 kernels | no stored port yet | PASS |

(n/c = CPU wall-clock time, not comparable to GPU kernel time.)

## CLI

Every MCP tool has a CLI equivalent, so you can drive the workflow by hand or from CI:

| command | MCP tool | what it does |
|---|---|---|
| `kernelport doctor` | `kernelport_doctor` | OS, GPU, extras, each backend with a reason |
| `kernelport scan [path]` | `scan_kernels` | find `__global__` kernels (also CUDA strings in .py) |
| `kernelport read <id>` | `read_kernel` | source, parsed signature, suggested input spec |
| `kernelport skill [name]` | `read_skill` | workflow, common_bugs, or a backend guide |
| `kernelport write-reference` | `write_reference` | save a candidate reference + spec |
| `kernelport validate` | `validate_reference` | original CUDA vs reference; freeze on PASS |
| `kernelport write-port` | `write_port` | save a port |
| `kernelport verify` | `verify_port` | port vs frozen reference |
| `kernelport bench` | `bench_port` | time port vs original (after PASS) |
| `kernelport status` | `list_status` | kernel × backend × hardware table |
| `kernelport report [--md]` | `make_report` | HTML or Markdown report |
| `kernelport setup` | – | register with Claude Code / Codex / Cursor / Gemini CLI |
| `kernelport mcp` | – | run the MCP server on stdio |

## Development

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -e ".[cuda,triton,cpu,dev]"
python -m pytest -q        # ~4 min; GPU tests skip without CUDA, cpu tests need only numba
```

## License

MIT
