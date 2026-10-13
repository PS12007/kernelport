"""Triton port of vector_add: out[i] = a[i] + b[i]."""

import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Each program handles one BLOCK-sized chunk (CUDA: one thread block).
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n  # CUDA: if (i < n)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


def run(a, b, out, n):
    BLOCK = 1024
    _vector_add_kernel[(triton.cdiv(n, BLOCK),)](a, b, out, n, BLOCK=BLOCK)
