import pytest

from conftest import needs_torch

pytestmark = needs_torch


def _cmp(actual, expected, dtype=None):
    from kernelport.harness.compare import compare_tensor
    from kernelport.harness.tolerances import tolerance_for

    dtype = dtype or actual.dtype
    return compare_tensor("y", actual, expected, dtype, tolerance_for(str(dtype).replace("torch.", "")))


def test_exact_match_passes():
    import torch

    x = torch.randn(100)
    assert _cmp(x, x.double()).passed


def test_within_tolerance_passes():
    import torch

    e = torch.ones(10, dtype=torch.float64)
    assert _cmp((e * (1 + 1e-6)).float(), e).passed


def test_outside_tolerance_reports_worst():
    import torch

    e = torch.zeros(20, dtype=torch.float64)
    a = torch.zeros(20)
    a[13] = 0.5
    r = _cmp(a, e)
    assert not r.passed
    assert r.worst_index == [13]
    assert r.max_abs_err == pytest.approx(0.5)
    assert 0.5 in r.actual_slice
    assert "outside tolerance" in r.reason


def test_shape_mismatch():
    import torch

    r = _cmp(torch.zeros(3), torch.zeros(4, dtype=torch.float64))
    assert not r.passed and "shape" in r.reason


def test_dtype_mismatch():
    import torch

    r = _cmp(torch.zeros(3), torch.zeros(3, dtype=torch.float64), dtype=torch.float16)
    assert not r.passed and "dtype" in r.reason


def test_unexpected_nan():
    import torch

    a = torch.zeros(5)
    a[2] = float("nan")
    r = _cmp(a, torch.zeros(5, dtype=torch.float64))
    assert not r.passed and "unexpected NaN" in r.reason and r.worst_index == [2]


def test_missing_nan():
    import torch

    e = torch.zeros(5, dtype=torch.float64)
    e[1] = float("nan")
    r = _cmp(torch.zeros(5), e)
    assert not r.passed and "missing NaN" in r.reason


def test_inf_sign_mismatch():
    import torch

    e = torch.tensor([float("inf")], dtype=torch.float64)
    r = _cmp(torch.tensor([float("-inf")]), e)
    assert not r.passed and "Inf" in r.reason


def test_matching_inf_passes():
    import torch

    e = torch.tensor([float("inf"), 1.0], dtype=torch.float64)
    assert _cmp(torch.tensor([float("inf"), 1.0]), e).passed


def test_reference_rounded_to_output_dtype():
    import torch

    # 70000 overflows fp16, so an fp16 kernel producing inf is correct.
    e = torch.tensor([70000.0], dtype=torch.float64)
    assert _cmp(torch.tensor([float("inf")], dtype=torch.float16), e).passed


def test_integers_must_match_exactly():
    import torch

    e = torch.arange(10)
    a = e.clone()
    a[7] += 1
    r = _cmp(a.to(torch.int32), e, dtype=torch.int32)
    assert not r.passed and r.worst_index == [7]
    assert _cmp(e.to(torch.int32), e, dtype=torch.int32).passed


def test_2d_worst_index():
    import torch

    e = torch.zeros(3, 4, dtype=torch.float64)
    a = torch.zeros(3, 4)
    a[2, 1] = 9.0
    assert _cmp(a, e).worst_index == [2, 1]
