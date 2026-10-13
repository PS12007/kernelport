"""Triton port of relu: y[i] = max(x[i], 0)."""

import triton
import triton.language as tl


@triton.jit
def _relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Same select as the CUDA kernel: x > 0 ? x : 0 (keeps the input dtype).
    y = tl.where(x > 0, x, 0.0).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def run(x, y, n):
    BLOCK = 1024
    _relu_kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
