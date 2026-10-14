# Installation

kernelport's core (MCP server, CLI, analyzer, reports, setup) needs only `mcp`, `typer`,
`rich` and `numpy`. Everything that runs kernels is an extra:

| extra | installs | needed for |
|---|---|---|
| `ref` | torch | running references (any machine, CPU torch is enough) |
| `cuda` | torch, cupy-cuda12x, NVRTC + CUDA runtime wheels | validating references against the original CUDA kernel |
| `triton` | torch, triton (Linux) / triton-windows (Windows) | the `triton` backend |
| `cpu` | torch, numba | the `cpu` backend |
| `metal` | torch, mlx (macOS only) | the `metal` backend (Apple Silicon) |
| `webgpu` | torch, wgpu | the `webgpu` backend (any Vulkan / Metal / D3D12 GPU) |
| `hip` | torch | the `hip` backend; needs ROCm builds of torch and CuPy installed first (not yet verified on AMD) |
| `agent` | anthropic | the optional built-in `kernelport agent` |
| `dev` | pytest | running the test suite |

## NVIDIA machine (validate + Triton + CPU)

PyPI's default torch wheel on Windows is CPU-only, and extras cannot pick the CUDA build.
Install torch from the PyTorch index **first**:

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows (PowerShell: .venv\Scripts\Activate.ps1)
source .venv/bin/activate                  # Linux / macOS

pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install "kernelport[cuda,triton,cpu] @ git+https://github.com/PS12007/kernelport"
kernelport doctor
```

Notes:

- **No CUDA toolkit, nvcc or Visual Studio is needed.** Originals are compiled with NVRTC from
  the `nvidia-cuda-nvrtc-cu12` wheel and launched with CuPy.
- `cupy-cuda12x[ctk]` is deliberately not used: it pulls about 1 GB of cuBLAS/cuFFT/cuSOLVER.
- On Windows the triton-windows minor version must match torch (torch 2.11 → triton-windows 3.7.x).
- New GPUs (e.g. Blackwell, sm_120) need torch built for CUDA 12.8 or later (`cu128`).

## Mac (Apple Silicon)

```bash
pip install "kernelport[metal,webgpu,cpu] @ git+https://github.com/PS12007/kernelport"
kernelport doctor
```

The default PyPI torch is fine here (references run on the CPU). See the next section for
the one step a Mac cannot do.

## No NVIDIA GPU (Mac, AMD, CPU-only): freeze once, verify anywhere

Ports can be verified on any machine that runs their backend, but **validating a reference
runs the original CUDA kernel, so it needs an NVIDIA GPU**. You only need one once per
reference:

1. On your machine, let the agent write the references and input specs (`write_reference`).
   `validate_reference` will answer `UNAVAILABLE`; that is expected.
2. Commit `.kernelport/` and run `kernelport validate --all` on any NVIDIA machine. A free
   Colab T4 works: open
   [`notebooks/freeze_references_colab.ipynb`](../notebooks/freeze_references_colab.ipynb)
   in Colab, set your repo URL, run all cells, and download the frozen `.kernelport/`.
3. Put the frozen `.kernelport/` back in your repo. Now `verify_port` works on your machine,
   and the agent can port and verify there.

Frozen references are plain files with a sha256 manifest, so they travel through git. Hashes
ignore line endings, so a Windows checkout with `core.autocrlf` does not break them, but any
other edit does: the reference then has to be validated again.

```bash
# CPU-only example: verify the cpu ports against references frozen elsewhere
pip install torch numba
pip install "kernelport[cpu] @ git+https://github.com/PS12007/kernelport"
kernelport verify --backend cpu
```

## Connecting your agent

```bash
kernelport setup              # detects Claude Code, Codex CLI, Cursor, Gemini CLI
kernelport setup --print      # only print the config snippets, change nothing
kernelport setup --uninstall  # remove exactly what setup added
```

`setup` shows every change before applying it, asks for approval, backs up each file it touches
to `~/.kernelport/backups/<timestamp>/`, and records what it added in
`~/.kernelport/setup_state.json` so `--uninstall` can undo it precisely. For Claude Code it uses
`claude mcp add --scope user` when the `claude` CLI is on PATH.

The server command is the interpreter kernelport is installed in (`<python> -m kernelport mcp`),
so the agent uses the same environment that `kernelport doctor` checked.

## Checking the install

`kernelport doctor` reports the OS, GPU and compute capability, which extras are installed,
and for each backend whether it is available. It also runs a tiny smoke kernel and gives a
reason when a backend is unavailable. `kernelport doctor --json` prints the same report as JSON.
