"""Triton port of saxpy, in place: y = a * x + y, computed in float32."""

import triton
import triton.language as tl


@triton.jit
def _saxpy_kernel(x_ptr, y_ptr, n, a, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)  # y is inout: read before writing
    tl.store(y_ptr + offs, a * x + y, mask=mask)


def run(n, a, x, y):
    BLOCK = 1024
    _saxpy_kernel[(triton.cdiv(n, BLOCK),)](x, y, n, a, BLOCK=BLOCK)
