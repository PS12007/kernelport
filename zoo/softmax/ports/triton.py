"""Triton port of row-wise softmax over a [rows, cols] row-major matrix.

One program per row (CUDA: one block per row). Two passes over the row in
BLOCK-sized chunks, so any number of columns works:
  1) online max and sum of exp(x - max), in float32
  2) write exp(x - max) / sum
"""

import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)  # int64 so row * cols cannot overflow
    x_row = x_ptr + row * cols
    y_row = y_ptr + row * cols

    # Pass 1: running max m and running sum s (rescaled when m grows).
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # Lanes that have only seen -inf keep s = 0 (avoid inf - inf = nan).
        scale = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
        s = s * scale + tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        m = m_new
    row_max = tl.max(m, axis=0)
    # Combine per-lane sums at the common row max.
    row_sum = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - row_max)), axis=0)
    inv = 1.0 / row_sum

    # Pass 2: normalise and store (cast back to the output dtype on store).
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + offs, tl.exp(x - row_max) * inv, mask=mask)


def run(x, y, rows, cols):
    BLOCK = min(triton.next_power_of_2(cols), 1024)
    _softmax_kernel[(rows,)](x, y, cols, BLOCK=BLOCK)
