import pytest

from kernelport.harness.expr import evaluate, evaluate_int


def test_arithmetic_and_helpers():
    dims = {"n": 1000, "rows": 3}
    assert evaluate_int("cdiv(n, 256)", dims) == 4
    assert evaluate_int("rows * n + 1", dims) == 3001
    assert evaluate_int("next_pow2(n)", dims) == 1024
    assert evaluate_int("min(n, 64)", dims) == 64
    assert evaluate(7, dims) == 7


@pytest.mark.parametrize("bad", ["__import__('os')", "n.__class__", "open('x')", "[1,2]", "lambda: 1"])
def test_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        evaluate(bad, {"n": 1})


def test_unknown_dim():
    with pytest.raises(ValueError, match="unknown dimension"):
        evaluate("m + 1", {"n": 1})


def test_non_integer():
    with pytest.raises(ValueError):
        evaluate_int("n / 3", {"n": 10})
