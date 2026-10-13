"""BUG: grid uses floor division, so the last partial block is never launched."""
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)


def run(a, b, out, n):
    if n // 256:
        _k[(n // 256,)](a, b, out, n, BLOCK=256)
