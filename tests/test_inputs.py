"""Inputs must be bit-identical on every machine. Golden hashes pin that."""

import pytest

from conftest import ZOO, needs_torch

pytestmark = needs_torch

# Recorded on Windows x86-64 (2026-10-13). Must match on Linux/macOS/ARM too.
GOLDEN = {
    ("softmax", "tiny_3x7", "float32"): "fd21348b369d3618aa387529866948eed22b9944def5daf0b5a47ea1a49faf99",
    ("softmax", "tiny_3x7", "float16"): "0d7d66d583858cbda2d49d29b7129a3d3aa160257ccc559cc894220c8525c570",
    ("softmax", "odd_5x1023", "bfloat16"): "a342d299ad5177b6a7d677134022b20cb5aed4dc01419cef62dd8d18a501aab1",
}


def _case(kernel, case, dtype):
    from kernelport.harness.inputs import make_case
    from kernelport.harness.spec import LaunchSpec

    spec = LaunchSpec.load(ZOO / kernel / "launch.json")
    return make_case(spec, spec.get_case(case), dtype)


@pytest.mark.parametrize("key", list(GOLDEN))
def test_golden_hashes(key):
    assert _case(*key).input_hash == GOLDEN[key]


def test_regeneration_is_identical():
    a = _case("vector_add", "odd_4097", "float32")
    b = _case("vector_add", "odd_4097", "float32")
    assert a.input_hash == b.input_hash
    for k in a.tensors:
        assert a.tensors[k].numpy().tobytes() == b.tensors[k].numpy().tobytes()


def test_dtypes_share_values():
    import torch

    f32 = _case("vector_add", "odd_1000", "float32").tensors["a"]
    f16 = _case("vector_add", "odd_1000", "float16").tensors["a"]
    assert torch.equal(f32.to(torch.float16), f16)


def test_outputs_prefilled_with_nan_and_edges_present():
    import torch

    d = _case("relu", "odd_1000", "float32")
    assert torch.isnan(d.tensors["y"]).all()
    assert d.tensors["x"][:4].tolist() == [0.0, -0.0, 1.0, -1.0]
    assert d.tensors["x"][4] == 1000.0


def test_different_cases_differ():
    assert _case("relu", "odd_1000", "float32").input_hash != _case("relu", "odd_1023", "float32").input_hash
