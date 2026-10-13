"""Triton port of cumsum: inclusive prefix sum along each row of [rows, cols].

One program per row (CUDA: one block per row). Like the kernel, the row is
scanned in BLOCK chunks: tl.cumsum scans a chunk (replacing the Hillis-Steele
shared-memory scan), and a float32 carry adds the totals of earlier chunks.
"""

import triton
import triton.language as tl


@triton.jit
def _cumsum_kernel(x_ptr, y_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * cols
    y_row = y_ptr + row * cols
    carry = 0.0
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + offs, carry + tl.cumsum(x, axis=0), mask=mask)
        carry += tl.sum(x, axis=0)


def run(x, y, rows, cols):
    BLOCK = min(triton.next_power_of_2(cols), 1024)
    _cumsum_kernel[(rows,)](x, y, cols, BLOCK=BLOCK)
