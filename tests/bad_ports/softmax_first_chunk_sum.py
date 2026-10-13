"""BUG: the row sum only covers the first 1024 columns. Correct whenever cols <= 1024."""
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row, y_row = x_ptr + row * cols, y_ptr + row * cols
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = tl.maximum(m, tl.load(x_row + offs, mask=offs < cols, other=float("-inf")).to(tl.float32))
    row_max = tl.max(m, axis=0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_row + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
    row_sum = tl.sum(tl.exp(x - row_max), axis=0)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=0.0).to(tl.float32)
        tl.store(y_row + offs, tl.exp(x - row_max) / row_sum, mask=offs < cols)


def run(x, y, rows, cols):
    _k[(rows,)](x, y, cols, BLOCK=min(triton.next_power_of_2(cols), 1024))
