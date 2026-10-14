"""Device-only slicing of real-world .cu files (host code, host headers, local includes)."""

import json
import shutil
from pathlib import Path

import pytest

from conftest import ROOT, needs_cuda
from kernelport.harness.device_source import SliceError, device_source

FIX = Path(__file__).parent / "fixtures" / "host_mixed"
LLMC = ROOT / "examples" / "llm.c"


def _body(src: str, name: str) -> str:
    """The sliced lines of one file, placed at their original line numbers (via #line markers)."""
    lines: dict[int, str] = {}
    current, lineno = None, 0
    for line in src.split("\n"):
        if line.startswith("#line "):
            _, n, fname = line.split(" ", 2)
            current, lineno = fname.strip('"'), int(n)
            continue
        if current == name:
            lines[lineno] = line
        lineno += 1
    return "\n".join(lines.get(i, "") for i in range(1, max(lines) + 1))


def test_host_code_dropped_device_code_kept():
    src = device_source(FIX / "kernels.cu")
    main = _body(src, "kernels.cu")
    for kept in ("__global__ void scale_kernel", "__global__ void pair_kernel", "__global__ void unrolled_kernel",
                 "__constant__ float bias_table", "typedef float (*unary_fn)(float);", "using index_t = int;",
                 "const int kBlock = 256;", "struct alignas(8) Pair", "enum class Mode", "namespace detail {",
                 "__device__ float clamp_max(float x) { return fminf", "__device__ float clamp_max(float x);",
                 '#define SCALE (M_PI / 4.0f)', 'extern "C" {'):
        assert kept in main, kept
    for dropped in ("int main(", "cudaMalloc", "printf", "void cpu_reference", "std::vector<int> host_sizes",
                    "int host_counter", "host_in_namespace", "host_in_extern_c", "#include <stdio.h>",
                    "#include <vector>", '#include "helpers.cuh"'):
        assert dropped not in main, dropped


def test_line_numbers_preserved():
    original = (FIX / "kernels.cu").read_text().splitlines()
    sliced = _body(device_source(FIX / "kernels.cu"), "kernels.cu").split("\n")
    for i, line in enumerate(original):
        if "__global__" in line:
            assert sliced[i] == line, (i, line)


def test_local_include_inlined_and_sliced():
    src = device_source(FIX / "kernels.cu")
    inc = _body(src, "helpers.cuh")
    assert "DEVICE_INLINE float twice" in inc and "__host__ __device__ T cdiv" in inc
    for dropped in ("cublasHandle_t", "workspace_size", "host_only_helper", "#include <cublas_v2.h>"):
        assert dropped not in inc, dropped
    assert '#line 7 "kernels.cu"' in src            # back in the main file after the include
    assert "#include <cuda_fp16.h>" in src           # cuda_runtime.h was dropped: half types come back
    assert src.count('#line 1 "helpers.cuh"') == 1


def test_include_outside_repo_refused(tmp_path):
    (tmp_path / "secret.h").write_text("__device__ int x;\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "k.cu").write_text('#include "../secret.h"\n__global__ void k() {}\n')
    with pytest.raises(SliceError, match="outside the repo"):
        device_source(repo / "k.cu", root=repo)


@needs_cuda
def test_sliced_fixture_compiles_with_nvrtc():
    import cupy as cp

    from kernelport.harness.run_cuda import NVRTC_OPTIONS, NVRTC_PRELUDE

    mod = cp.RawModule(code=NVRTC_PRELUDE + device_source(FIX / "kernels.cu"), options=NVRTC_OPTIONS,
                       name_expressions=["scale_kernel", "pair_kernel", "unrolled_kernel<4>"])
    for n in ("scale_kernel", "pair_kernel", "unrolled_kernel<4>"):
        mod.get_function(n)


@needs_cuda
def test_llmc_originals_validate_from_unmodified_files(tmp_path):
    """examples/llm.c: the frozen references still match the unmodified llm.c kernels."""
    from kernelport import tools
    from kernelport.workspace import Workspace

    repo = tmp_path / "llm.c"
    shutil.copytree(LLMC, repo)
    ws = Workspace(repo)
    ids = ws.kernel_ids()
    frozen = [k for k in ids if ws.is_frozen(k)]
    assert len(frozen) == 5
    for kid in frozen:
        d = ws.get(kid).dir
        # Validate the frozen copy again: a passing re-validation of the same reference.
        (d / "reference.py").write_bytes((d / "frozen" / "reference.py").read_bytes())
        (d / "launch.json").write_bytes((d / "frozen" / "launch.json").read_bytes())
        res = tools.validate_reference(kid, root=str(repo))
        assert res["status"] == "PASS", (kid, res.get("report"))
        assert json.loads((d / "frozen" / "manifest.json").read_text())["kernel_id"] == kid
