"""BUG: leaky relu (0.01 * x for negatives) instead of relu."""
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.where(x > 0, x, x * 0.01).to(x.dtype), mask=mask)


def run(x, y, n):
    _k[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
