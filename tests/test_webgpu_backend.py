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


def _cpu_adapter() -> bool:
    from kernelport.backends.webgpu import runner

    info = runner.device()._kp_adapter_info
    return str(info.get("adapter_type", "")).upper() == "CPU" or "llvmpipe" in str(info.get("device", "")).lower()


def _verify(kernel, port):
    from kernelport.zoo import verify_zoo_port

    return verify_zoo_port(ZOO / kernel, port, backend="webgpu")


def test_every_zoo_kernel_has_a_webgpu_port():
    assert len(KERNELS) == 12


def test_float_literals_browsers_reject_are_refused():
    """No GPU needed. naga accepts -3.4028235e38 (above FLT_MAX); Chrome/Edge's Tint does not."""
    from kernelport.backends.webgpu.runner import check_browser_literals

    for ok in ("var m = -3.4028234e38;", "let a = 1.0; let b = 2e10f; let c = 6.5e4h; let d = 1e-45;",
               "// var m = -3.4028235e38;\n", "/* 1e39 */ let x = .5;", "let i = 123456789012345u;"):
        check_browser_literals(ok)
    for bad, line in (("var m = -3.4028235e38;", 1), ("\nlet x = 1e39f;", 2), ("let h = 7e4h;", 1)):
        with pytest.raises(ValueError, match=f"line {line}: .*browsers"):
            check_browser_literals(bad)
    for kernel in KERNELS:  # every zoo port is browser-safe
        from kernelport.backends.webgpu.runner import load_port
        check_browser_literals(load_port(ZOO / kernel / "ports" / "webgpu.py").WGSL)


def test_uninitialised_loop_vars_are_refused():
    """No GPU needed. wgpu-native zeroes a loop-local `var x: T;` once, WGSL (and browsers) every iteration."""
    from kernelport.backends.webgpu.runner import check_loop_vars

    for ok in ("var acc: array<f32, 128>;\nfor (var i = 0u; i < 3u; i++) { acc[i] = 1.0; }",
               "for (var i = 0u; i < 3u; i++) { var s = 0.0; var a = array<f32, 4>(); }",
               "var<workgroup> red: array<f32, 256>;\nfn f() { loop { break; } }",
               "for (var i = 0u; i < 3u; i++) { // var s: f32;\n }"):
        check_loop_vars(ok)
    for bad, suggestion in (("for (var i = 0u; i < 3u; i++) {\n  var s: f32;\n}", "var s = f32();"),
                            ("loop { if (x) { var a: array<f32, 4>; } break; }", "var a = array<f32, 4>();"),
                            ("while (c) { var v : vec4<f32> ; }", "var v = vec4<f32>();")):
        with pytest.raises(ValueError, match="inside a loop") as e:
            check_loop_vars(bad)
        assert suggestion in str(e.value)


def test_binding_size_checks():
    """No GPU needed: limits are passed in."""
    from kernelport.backends.webgpu import runner
    from kernelport.harness.inputs import make_case
    from kernelport.harness.results import CaseNotRunnable
    from kernelport.harness.spec import LaunchSpec

    spec = LaunchSpec.load(ZOO / "embedding_gather" / "launch.json")
    data = make_case(spec, spec.get_case(spec.bench_case), "float32")
    sizes = {p.name: data.tensors[p.name].numel() * data.tensors[p.name].element_size()
             for p in spec.params if p.is_tensor}
    biggest, total = max(sizes.values()), sum(sizes.values())
    runner._check_binding_sizes(spec, data, biggest, None)          # fits exactly
    runner._check_binding_sizes(spec, data, biggest, total)
    with pytest.raises(CaseNotRunnable, match=r"needs a \d+ MiB storage binding; this adapter allows 128 MiB"):
        runner._check_binding_sizes(spec, data, 128 * 2**20, None)
    with pytest.raises(CaseNotRunnable, match="in total"):
        runner._check_binding_sizes(spec, data, biggest, total - 1)


@pytest.mark.skipif(not _has("torch"), reason="needs torch")
def test_not_runnable_cases_are_named_never_passed():
    import torch

    from kernelport.harness.results import CaseNotRunnable
    from kernelport.harness.spec import LaunchSpec
    from kernelport.harness.verify import run_cases

    spec = LaunchSpec.load(ZOO / "relu" / "launch.json")
    big = spec.cases[-1].name

    def runner(data, refuse=lambda name: False):
        if refuse(data.case):
            raise CaseNotRunnable("x needs a 512 MiB storage binding; this adapter allows 128 MiB")
        return {"y": torch.clamp(data.tensors["x"], min=0)}

    def run(refuse):
        return run_cases(kernel="relu", backend="webgpu", hardware="test", spec=spec,
                         reference_path=ZOO / "relu" / "reference.py", code_hash="h",
                         runner=lambda d: runner(d, refuse), supported_dtypes=("float32",))

    r = run(lambda name: name == big)
    assert r.status == "PASS", r.failure_report()
    assert f"cannot run, not tested: {big}/float32 (x needs a 512 MiB" in r.reason
    assert big not in {c.case for c in r.cases}
    assert run(lambda name: True).status == "SKIPPED"


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
    if "racy" in port and r.status == "PASS" and _cpu_adapter():
        pytest.skip("the lost-update race did not happen: a CPU adapter (llvmpipe) may run workgroups one by one")
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
