"""Triton port of llm.c encoder_forward_kernel3: out[b, t, c] = wte[inp[b, t], c] + wpe[t, c].

Element-wise over the flattened [B*T*C] output like the original (which packs 8 bf16 per thread);
fp32 add, cast back to bf16. Any C works (the original needs C % 8 == 0).
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TUNE = {"BLOCK": [512, 1024, 2048, 4096], "NUM_WARPS": [2, 4, 8]}


@triton.jit
def _encoder(out_ptr, inp_ptr, wte_ptr, wpe_ptr, n, T, C, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    bt = offs // C
    c = offs % C
    ix = tl.load(inp_ptr + bt, mask=mask, other=0).to(tl.int64)
    e = tl.load(wte_ptr + ix * C + c, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(wpe_ptr + (bt % T) * C + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (e + p).to(out_ptr.dtype.element_ty), mask=mask)


def run(out, inp, wte, wpe, B, T, C):
    n = B * T * C
    _encoder[(triton.cdiv(n, BLOCK),)](out, inp, wte, wpe, n, T, C, BLOCK=BLOCK, num_warps=NUM_WARPS)
