"""Triton port of llm.c residual_forward_kernel1: out = bf16(float(a) + float(b))."""

import triton
import triton.language as tl


@triton.jit
def _residual(out_ptr, a_ptr, b_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (a + b).to(out_ptr.dtype.element_ty), mask=mask)


def run(out, inp1, inp2, N):
    BLOCK = 1024
    _residual[(triton.cdiv(N, BLOCK),)](out, inp1, inp2, N, BLOCK=BLOCK)
