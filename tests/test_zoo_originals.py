"""Every zoo original must pass its reference, and subtly wrong references must not.

The mutant references below are each a plausible mistake an agent could make
when writing a reference. If validation accepted any of them, a wrong
reference could be frozen and every port would be checked against it.
"""

import pytest

from conftest import ZOO, needs_cuda
from kernelport.zoo import list_kernels

KERNELS = list_kernels(ZOO)

MUTANTS = {
    "vector_add": "def reference(a, b):\n    return a + b * 1.001\n",
    "relu": "def reference(x):\n    return x.clamp(min=0.0, max=999.0)\n",
    "saxpy": "def reference(a, x, y):\n    return {'y': int(a) * x + y}\n",
    "transpose": "def reference(inp, rows, cols):\n    return inp.reshape(cols, rows)\n",
    "reduce_sum": "def reference(x, out):\n    return {'out': out + x[:-1].sum()}\n",
    "softmax": "import torch\ndef reference(x):\n    return torch.softmax(x, dim=0)\n",
    "layernorm": (
        "import torch\n"
        "def reference(x, gamma, beta, eps):\n"
        "    mean = x.mean(-1, keepdim=True)\n"
        "    var = x.var(-1, keepdim=True, unbiased=True).nan_to_num(0.0)  # wrong: unbiased\n"
        "    return (x - mean) / torch.sqrt(var + eps) * gamma + beta\n"
    ),
    "rmsnorm": "import torch\ndef reference(x, weight, eps):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n",
    "matmul_naive": "def reference(A, B):\n    return A[:, :-1] @ B[:-1]\n",
    "rope": (
        "import torch\n"
        "def reference(positions, x, cos_cache, sin_cache):\n"
        "    c = cos_cache[positions][:, None, :]; s = sin_cache[positions][:, None, :]\n"
        "    x0, x1 = x[..., 0::2], x[..., 1::2]\n"
        "    y = torch.empty_like(x)\n"
        "    y[..., 0::2] = x0 * c + x1 * s  # wrong sign\n"
        "    y[..., 1::2] = -x0 * s + x1 * c\n"
        "    return y\n"
    ),
    "embedding_gather": "def reference(ids, table):\n    return table[ids.clamp(max=table.shape[0] - 2)]\n",
    "cumsum": "import torch\ndef reference(x):\n    return torch.cumsum(x, dim=-1) - x  # exclusive scan\n",
}


def test_every_zoo_kernel_has_a_mutant():
    assert sorted(MUTANTS) == sorted(KERNELS)


def test_zoo_layout():
    for k in KERNELS:
        for f in ("kernel.cu", "launch.json", "reference.py"):
            assert (ZOO / k / f).is_file(), f"zoo/{k}/{f} missing"


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
def test_original_passes_reference(kernel):
    from kernelport.zoo import validate_original

    r = validate_original(ZOO / kernel, "test")
    assert r.status == "PASS", r.failure_report()


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
def test_mutant_reference_fails(kernel, tmp_path):
    from kernelport.zoo import validate_original

    ref = tmp_path / "reference.py"
    ref.write_text(MUTANTS[kernel])
    r = validate_original(ZOO / kernel, "test", reference=ref, stop_after_failures=1)
    assert r.status == "FAIL", f"mutant reference for {kernel} was accepted"
