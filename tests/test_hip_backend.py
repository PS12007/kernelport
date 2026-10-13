"""hip backend.

On an AMD GPU with ROCm torch + ROCm CuPy, every zoo hip port is verified for real.
Elsewhere, the runner's plumbing (entry naming, argument binding, launch, fetch) is
exercised on NVIDIA by compiling the HIP ports with NVRTC after mapping the HIP headers
and bf16 type name back to CUDA's. That proves the runner and the HIP sources' logic,
not that hipRTC accepts them: only a ROCm run can show that.
"""

from pathlib import Path

import pytest

from conftest import ZOO, needs_cuda
from kernelport.backends.base import get_backend

KERNELS = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/hip.hip"))
on_rocm = get_backend("hip").is_available()[0]


def test_every_zoo_kernel_has_a_hip_port():
    assert len(KERNELS) == 12
    for k in KERNELS:
        src = (ZOO / k / "ports" / "hip.hip").read_text()
        assert "cuda_fp16.h" not in src and "cuda_bf16.h" not in src
        assert "__shfl" not in src  # warp-size independent


@pytest.mark.skipif(not on_rocm, reason="needs ROCm torch + ROCm CuPy on an AMD GPU")
@pytest.mark.parametrize("kernel", KERNELS)
def test_zoo_hip_port_passes_on_rocm(kernel):
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / kernel, ZOO / kernel / "ports" / "hip.hip", backend="hip")
    assert r.status == "PASS", r.failure_report()


@needs_cuda
@pytest.mark.parametrize("kernel", ["vector_add", "rmsnorm", "reduce_sum", "softmax"])
def test_hip_runner_plumbing_on_nvidia(kernel, tmp_path, monkeypatch):
    from kernelport.backends.hip import runner
    from kernelport.harness.spec import LaunchSpec
    from kernelport.harness.verify import run_cases

    src = (ZOO / kernel / "ports" / "hip.hip").read_text()
    src = src.replace("#include <hip/hip_fp16.h>", "#include <cuda_fp16.h>")
    src = src.replace("#include <hip/hip_bf16.h>", "#include <cuda_bf16.h>")
    port = tmp_path / "port.hip"
    port.write_text(src)
    monkeypatch.setitem(runner.HIP_TYPE_NAMES, "bfloat16", "__nv_bfloat16")

    spec = LaunchSpec.load(ZOO / kernel / "launch.json")

    def run(data):
        p = runner.prepare(Path(port), spec, data)
        p.launch()
        p.sync()
        return p.fetch()

    r = run_cases(kernel=kernel, backend="hip", hardware="nvidia-plumbing", spec=spec,
                  reference_path=ZOO / kernel / "reference.py", runner=run, code_hash="x")
    assert r.status == "PASS", r.failure_report()


def test_hip_type_names_use_hip_spelling():
    from kernelport.backends.hip.runner import HIP_TYPE_NAMES

    assert HIP_TYPE_NAMES["bfloat16"] == "__hip_bfloat16" and HIP_TYPE_NAMES["float16"] == "__half"
