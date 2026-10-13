"""Triton port of layernorm over the last dim of a [rows, cols] matrix.

One program per row (CUDA: one block per row). Same two-pass math as the
kernel, in float32: mean, then biased variance of x - mean, then normalise.
Rows longer than BLOCK are processed in chunks.
"""

import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, g_ptr, b_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * cols
    y_row = y_ptr + row * cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_row + offs, mask=offs < cols, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        d = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32) - mean
        acc += tl.where(mask, d * d, 0.0)  # masked lanes would add mean^2
    rstd = 1.0 / tl.sqrt_rn(tl.sum(acc, axis=0) / cols + eps)

    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + offs, (x - mean) * rstd * g + b, mask=mask)


def run(x, gamma, beta, y, rows, cols, eps):
    BLOCK = min(triton.next_power_of_2(cols), 1024)
    _layernorm_kernel[(rows,)](x, gamma, beta, y, cols, eps, BLOCK=BLOCK)
