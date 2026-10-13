"""Triton port of rope with interleaved pairs.

One program per token (CUDA: one block per token). Lanes cover the flattened
(head, pair) index j, like the kernel's thread stride loop, in BLOCK chunks:
  y[t, h, 2i]   = x0 * cos[p, i] - x1 * sin[p, i]
  y[t, h, 2i+1] = x0 * sin[p, i] + x1 * cos[p, i]
"""

import triton
import triton.language as tl


@triton.jit
def _rope_kernel(pos_ptr, x_ptr, cos_ptr, sin_ptr, y_ptr, heads, head_dim, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    half = head_dim // 2
    p = tl.load(pos_ptr + t)  # int64 position
    total = heads * half
    for start in range(0, total, BLOCK):
        j = start + tl.arange(0, BLOCK)
        mask = j < total
        h = j // half
        i = j % half
        base = (t * heads + h) * head_dim + 2 * i
        x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + p * half + i, mask=mask, other=0.0)
        s = tl.load(sin_ptr + p * half + i, mask=mask, other=0.0)
        tl.store(y_ptr + base, x0 * c - x1 * s, mask=mask)
        tl.store(y_ptr + base + 1, x0 * s + x1 * c, mask=mask)


def run(positions, x, cos_cache, sin_cache, y, tokens, heads, head_dim):
    BLOCK = min(triton.next_power_of_2(heads * (head_dim // 2)), 1024)
    _rope_kernel[(tokens,)](positions, x, cos_cache, sin_cache, y, heads, head_dim, BLOCK=BLOCK)
