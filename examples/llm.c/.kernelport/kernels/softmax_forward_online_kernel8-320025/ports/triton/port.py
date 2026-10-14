"""Triton port of llm.c softmax_forward_online_kernel8: row-wise softmax, online max/sum in fp32.

One program per row. Unlike the original, any C >= 1 works (the original needs C >= blockDim.x);
the frozen spec only tests C >= 256, the range where the original is correct.
"""

import triton
import triton.language as tl


@triton.jit
def _softmax(out_ptr, inp_ptr, C, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = inp_ptr + row * C
    y_row = out_ptr + row * C
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    s = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < C, other=float("-inf"))
        m_new = tl.maximum(m, x)
        scale = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
        s = s * scale + tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        m = m_new
    row_max = tl.max(m, axis=0)
    row_sum = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - row_max)), axis=0)
    for start in range(0, C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(x_row + offs, mask=mask, other=0.0)
        tl.store(y_row + offs, tl.exp(x - row_max) / row_sum, mask=mask)


def run(out, inp, N, C):
    BLOCK = min(triton.next_power_of_2(C), 1024)
    _softmax[(N,)](out, inp, C, BLOCK=BLOCK)
