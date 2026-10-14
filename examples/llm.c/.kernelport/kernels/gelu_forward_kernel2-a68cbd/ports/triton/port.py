"""Triton port of llm.c gelu_forward_kernel2 (bf16 in/out, fp32 math, tanh approximation)."""

import triton
import triton.language as tl

SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)  # jit code may only read constexpr globals


@triton.jit
def _gelu(out_ptr, inp_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    # tanh(z) = 1 - 2 / (exp(2z) + 1); saturates cleanly to +-1 for large |z|
    t = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)
    tl.store(out_ptr + offs, (0.5 * x * (1.0 + t)).to(out_ptr.dtype.element_ty), mask=mask)


def run(out, inp, N):
    BLOCK = 1024
    _gelu[(triton.cdiv(N, BLOCK),)](out, inp, N, BLOCK=BLOCK)
