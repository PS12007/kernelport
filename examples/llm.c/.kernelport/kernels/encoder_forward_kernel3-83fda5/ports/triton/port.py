"""Triton port of llm.c encoder_forward_kernel3: out[b, t, c] = wte[inp[b, t], c] + wpe[t, c].

One program per (token row, block of C): the token id is loaded once per row and every
load/store is contiguous, with no per-element division (the original packs 8 bf16 per thread).
fp32 add, cast back to bf16. Any C works (the original needs C % 8 == 0).
"""

import triton
import triton.language as tl

BLOCK_C = 1024
NUM_WARPS = 4
TUNE = {"BLOCK_C": [256, 512, 1024, 2048], "NUM_WARPS": [1, 2, 4, 8]}


@triton.jit
def _encoder(out_ptr, inp_ptr, wte_ptr, wpe_ptr, T, C, BLOCK_C: tl.constexpr):
    bt = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c < C
    ix = tl.load(inp_ptr + bt).to(tl.int64)
    t = bt % T
    e = tl.load(wte_ptr + ix * C + c, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(wpe_ptr + t.to(tl.int64) * C + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + bt.to(tl.int64) * C + c, (e + p).to(out_ptr.dtype.element_ty), mask=mask)


def run(out, inp, wte, wpe, B, T, C):
    block = min(BLOCK_C, triton.next_power_of_2(C))
    _encoder[(B * T, triton.cdiv(C, block))](out, inp, wte, wpe, T, C, BLOCK_C=block, num_warps=NUM_WARPS)
