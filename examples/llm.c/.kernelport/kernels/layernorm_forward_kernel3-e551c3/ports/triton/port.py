"""Triton port of llm.c layernorm_forward_kernel3: one program per row (CUDA: one warp per row).

Two passes over the row in fp32, like the original: mean, then mean of squared deviations.
"""

import triton
import triton.language as tl


@triton.jit
def _layernorm(out_ptr, mean_ptr, rstd_ptr, inp_ptr, w_ptr, b_ptr, C, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = inp_ptr + row * C
    o_row = out_ptr + row * C
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_row + offs, mask=offs < C, other=0.0)
    m = tl.sum(acc, axis=0) / C
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        d = tl.load(x_row + offs, mask=offs < C, other=0.0) - m
        acc += tl.where(offs < C, d * d, 0.0)
    s = 1.0 / tl.sqrt_rn(tl.sum(acc, axis=0) / C + 1e-5)
    tl.store(mean_ptr + row, m)
    tl.store(rstd_ptr + row, s)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(x_row + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(o_row + offs, s * (x - m) * w + b, mask=mask)


def run(out, mean, rstd, inp, weight, bias, N, C):
    BLOCK = min(triton.next_power_of_2(C), 1024)
    _layernorm[(N,)](out, mean, rstd, inp, weight, bias, C, BLOCK=BLOCK)
