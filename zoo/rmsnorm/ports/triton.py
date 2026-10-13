"""Triton port of rmsnorm over the last dim: y = x / sqrt(mean(x^2) + eps) * weight.

One program per row; the warp-shuffle + shared-memory reduction of the CUDA
kernel becomes a single tl.sum. float32 math, chunked for long rows.
"""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * cols
    y_row = y_ptr + row * cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < cols, other=0.0).to(tl.float32)
        acc += x * x
    inv_rms = 1.0 / tl.sqrt_rn(tl.sum(acc, axis=0) / cols + eps)

    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + offs, x * inv_rms * w, mask=mask)


def run(x, weight, y, rows, cols, eps):
    BLOCK = min(triton.next_power_of_2(cols), 1024)
    _rmsnorm_kernel[(rows,)](x, weight, y, cols, eps, BLOCK=BLOCK)
