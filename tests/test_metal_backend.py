"""metal backend.

On an Apple Silicon Mac with MLX, every zoo metal port is verified for real. Elsewhere,
a fake `mlx.core` records how the runner builds and calls `mx.fast.metal_kernel`
(names, inout naming, scalar arrays, template, grid, NaN init, bf16 round trip). That
proves the runner's contract, not that Metal compiles the ports: only a Mac run can.
"""

import math
import sys
import types

import numpy as np
import pytest
import torch

from conftest import ZOO
from kernelport.backends.base import get_backend

KERNELS = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/metal.py"))
on_mac = get_backend("metal").is_available()[0]


@pytest.fixture(autouse=True)
def _fresh_kernel_cache():
    # Kernels built with the fake mlx.core must never be reused by a real-Mac test.
    from kernelport.backends.metal import runner

    runner._kernels.clear()
    yield
    runner._kernels.clear()


class FakeArray:
    def __init__(self, a, dtype=None):
        self.a = np.asarray(a)
        self.dtype = dtype or str(self.a.dtype)

    def astype(self, dtype):
        return FakeArray(self.a.astype(np.float32), dtype)

    def __array__(self, dtype=None, copy=None):
        return self.a


def _fake_mlx(calls, compute):
    core = types.ModuleType("mlx.core")
    for n in ("float32", "float16", "bfloat16", "int32", "int64", "uint32", "int8", "uint8", "int16"):
        setattr(core, n, n)
    core.array = lambda a: FakeArray(a)
    core.eval = lambda *a: calls.append(("eval", len(a)))

    def metal_kernel(**kw):
        calls.append(("build", kw))

        def run(**call):
            calls.append(("call", call))
            return compute(call)
        return run
    core.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    core.metal = types.SimpleNamespace(is_available=lambda: True)
    pkg = types.ModuleType("mlx")
    pkg.core = core
    return pkg, core


def test_every_zoo_kernel_has_a_metal_port():
    assert len(KERNELS) == 12


def test_runner_builds_and_calls_metal_kernel(monkeypatch):
    from kernelport.backends.metal import runner
    from kernelport.harness.inputs import make_case
    from kernelport.harness.spec import LaunchSpec

    calls = []

    def compute(call):  # saxpy semantics on the fake "GPU": inputs are x, y, n, a
        x, y, n, a = (np.asarray(v) for v in call["inputs"])
        return [FakeArray((a[0] * x.astype(np.float32) + y.astype(np.float32)))]

    pkg, core = _fake_mlx(calls, compute)
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    runner._kernels.clear()

    spec = LaunchSpec.load(ZOO / "saxpy" / "launch.json")
    data = make_case(spec, spec.get_case("odd_1000"), "bfloat16")
    p = runner.prepare(ZOO / "saxpy" / "ports" / "metal.py", spec, data)
    p.launch()
    p.sync()
    out = p.fetch()

    build = next(c[1] for c in calls if c[0] == "build")
    assert build["input_names"] == ["x", "y", "n", "a"]          # inout y is an input, scalars last
    assert build["output_names"] == ["y_out"]                     # inout written as <name>_out
    assert build["atomic_outputs"] is False and build["ensure_row_contiguous"] is True
    call = next(c[1] for c in calls if c[0] == "call")
    assert call["template"] == [("T", "bfloat16")]
    assert call["grid"] == (1000, 1, 1) and call["threadgroup"] == (256, 1, 1)
    assert call["output_shapes"] == [(1000,)] and call["output_dtypes"] == ["bfloat16"]
    assert math.isnan(call["init_value"])                         # unwritten outputs show up as NaN
    assert call["inputs"][0].dtype == "bfloat16"                  # bf16 goes through float32, exactly
    assert np.asarray(call["inputs"][2]).tolist() == [1000]       # scalar n as a 1-element array
    assert out["y"].dtype == torch.bfloat16 and out["y"].shape == (1000,)
    assert ("eval", 1) in calls


def test_atomic_port_options_are_passed(monkeypatch):
    from kernelport.backends.metal import runner
    from kernelport.harness.inputs import make_case
    from kernelport.harness.spec import LaunchSpec

    calls = []
    pkg, core = _fake_mlx(calls, lambda call: [FakeArray(np.zeros(1, np.float32))])
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    runner._kernels.clear()
    spec = LaunchSpec.load(ZOO / "reduce_sum" / "launch.json")
    data = make_case(spec, spec.get_case("odd_4097"), "float32")
    p = runner.prepare(ZOO / "reduce_sum" / "ports" / "metal.py", spec, data)
    p.launch()
    build = next(c[1] for c in calls if c[0] == "build")
    call = next(c[1] for c in calls if c[0] == "call")
    assert build["atomic_outputs"] is True and build["output_names"] == ["out_out"]
    assert call["init_value"] == 0.0 and call["grid"] == (17 * 256, 1, 1)


@pytest.mark.skipif(not on_mac, reason="needs an Apple Silicon Mac with MLX")
@pytest.mark.parametrize("kernel", KERNELS)
def test_zoo_metal_port_passes_on_mac(kernel):
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / kernel, ZOO / kernel / "ports" / "metal.py", backend="metal")
    assert r.status == "PASS", r.failure_report()
