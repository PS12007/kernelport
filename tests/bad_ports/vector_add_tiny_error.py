"""BUG: adds a tiny relative error (5e-6) - only fp32 tolerance should catch it."""
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(a_ptr + offs, mask=mask).to(tl.float32) + tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, s * 1.000005, mask=mask)


def run(a, b, out, n):
    _k[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK=256)
