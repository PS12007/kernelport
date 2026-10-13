"""webgpu backend (WGSL via wgpu-native): zoo ports pass, broken ports fail for the right reason."""

from pathlib import Path

import pytest

from conftest import ZOO, _has
from kernelport.backends.base import get_backend

BAD = Path(__file__).parent / "bad_ports" / "webgpu"
KERNELS = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/webgpu.py"))


def _usable() -> bool:
    if not (_has("wgpu") and _has("torch")):
        return False
    return get_backend("webgpu").is_available()[0]


needs_webgpu = pytest.mark.skipif(not _usable(), reason="needs wgpu + a WebGPU adapter")


def _verify(kernel, port):
    from kernelport.zoo import verify_zoo_port

    return verify_zoo_port(ZOO / kernel, port, backend="webgpu")


def test_every_zoo_kernel_has_a_webgpu_port():
    assert len(KERNELS) == 12


def test_bindings_are_generated_from_the_spec():
    """Pure string generation: no GPU needed."""
    from unittest import mock

    from kernelport.backends.webgpu import runner
    from kernelport.harness.spec import LaunchSpec

    spec = LaunchSpec.load(ZOO / "rope" / "launch.json")
    with mock.patch.object(runner, "has_f16", return_value=True):
        src = runner.bindings_source(spec, "float16", {})
    assert "alias T = f16;" in src
    assert "@binding(0) var<storage, read> positions: array<vec2<u32>>;" in src
    assert "@binding(2) var<storage, read> cos_cache: array<f32>;" in src
    assert "@binding(4) var<storage, read_write> y: array<T>;" in src
    assert "struct Params { tokens: i32, heads: i32, head_dim: i32 };" in src
    assert "@binding(5) var<uniform> params: Params;" in src


@needs_webgpu
@pytest.mark.parametrize("kernel", KERNELS)
def test_zoo_webgpu_port_passes(kernel):
    r = _verify(kernel, ZOO / kernel / "ports" / "webgpu.py")
    assert r.status == "PASS", r.failure_report()
    assert "WebGPU/" in r.hardware


@needs_webgpu
@pytest.mark.parametrize("kernel,port,expect", [
    ("vector_add", "vector_add_no_stride.py", "NaN"),
    ("vector_add", "vector_add_grid_too_big.py", "at most 65535 workgroups"),
    ("relu", "relu_wrong_name.py", "output"),
    ("reduce_sum", "reduce_sum_racy_add.py", "outside tolerance"),
])
def test_bad_webgpu_port_fails(kernel, port, expect):
    r = _verify(kernel, BAD / port)
    assert r.status == "FAIL"
    assert expect in r.failure_report(), r.failure_report()


@needs_webgpu
def test_no_stride_bug_only_caught_by_large_case():
    r = _verify("vector_add", BAD / "vector_add_no_stride.py")
    assert {c.case for c in r.cases if not c.passed} == {"large_16M"}


@needs_webgpu
def test_webgpu_bench_not_comparable():
    from kernelport.harness.spec import LaunchSpec
    from kernelport.zoo import bench_port

    kdir = ZOO / "relu"
    spec = LaunchSpec.load(kdir / "launch.json")
    spec.bench_case = "odd_4097"
    b = bench_port(spec, kdir / spec.source, kdir / "ports" / "webgpu.py", "webgpu")
    assert b["method"] == "wall_clock" and b["port_ms"] > 0 and not b["comparable"]
