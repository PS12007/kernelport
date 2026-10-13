"""BUG: no max subtraction, so exp overflows on large inputs."""
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=float("-inf")).to(tl.float32)
    e = tl.exp(x)
    tl.store(y_ptr + row * cols + offs, e / tl.sum(e, axis=0), mask=mask)


def run(x, y, rows, cols):
    _k[(rows,)](x, y, cols, BLOCK=triton.next_power_of_2(cols))
