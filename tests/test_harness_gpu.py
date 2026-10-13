"""End-to-end: originals pass, good ports pass, deliberately broken ports FAIL.

If any bad port here ever passes, the harness is not trustworthy.
"""

import shutil
from pathlib import Path

import pytest

from conftest import ZOO, needs_cuda, needs_triton

BAD = Path(__file__).parent / "bad_ports"


@needs_cuda
def test_wrong_reference_fails_validation(tmp_path):
    from kernelport.zoo import validate_original

    kdir = tmp_path / "vector_add"
    shutil.copytree(ZOO / "vector_add", kdir)
    (kdir / "reference.py").write_text("def reference(a, b, n):\n    return a + 2 * b\n")
    assert validate_original(kdir, "test").status == "FAIL"


TRITON_PORTS = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/triton.py"))


def test_every_zoo_kernel_has_a_triton_port():
    assert len(TRITON_PORTS) == 12


@needs_triton
@pytest.mark.parametrize("kernel", TRITON_PORTS)
def test_good_triton_port_passes(kernel):
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / kernel, ZOO / kernel / "ports" / "triton.py")
    assert r.status == "PASS", r.failure_report()


@needs_triton
@pytest.mark.parametrize("kernel,port,expect", [
    ("vector_add", "vector_add_missing_tail.py", "NaN"),
    ("vector_add", "vector_add_sub.py", "outside tolerance"),
    ("vector_add", "vector_add_tiny_error.py", "outside tolerance"),
    ("vector_add", "vector_add_compile_error.py", "error"),
    ("vector_add", "vector_add_noop.py", "NaN"),
    ("relu", "relu_leaky.py", "outside tolerance"),
    ("softmax", "softmax_no_max.py", "NaN"),
    ("softmax", "softmax_first_chunk_sum.py", "outside tolerance"),
    ("matmul_naive", "matmul_tf32.py", "outside tolerance"),
])
def test_bad_port_fails(kernel, port, expect):
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / kernel, BAD / port)
    assert r.status == "FAIL"
    assert expect in r.failure_report(), r.failure_report()


@needs_triton
def test_first_chunk_bug_only_caught_by_wide_rows():
    """The bug is invisible for cols <= 1024; the odd/large cases must catch it."""
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / "softmax", BAD / "softmax_first_chunk_sum.py")
    by_case = {(c.case, c.dtype): c.passed for c in r.cases}
    assert by_case[("pow2_64x1024", "float32")] is True
    assert by_case[("odd_3x4097", "float32")] is False


@needs_triton
def test_tiny_error_caught_in_fp32():
    from kernelport.zoo import verify_zoo_port

    r = verify_zoo_port(ZOO / "vector_add", BAD / "vector_add_tiny_error.py")
    fp32 = [c.passed for c in r.cases if c.dtype == "float32" and c.case not in ("tiny_1",)]
    assert not any(fp32)
