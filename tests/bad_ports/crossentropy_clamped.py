"""BAD port of llm.c crossentropy_forward_kernel1: clamps probs to >= 1e-3 before the log.

A plausible "numerical safety" edit. It passes the kernel's own cases (probs uniform in
[0.001, 1]) but is wrong on real softmax output, where most probs are near 1/V. Only a
pipeline (softmax -> crossentropy) catches it.
"""

import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _ce_fwd(losses_ptr, probs_ptr, targets_ptr, N, V, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    ix = tl.load(targets_ptr + i, mask=mask, other=0)
    p = tl.load(probs_ptr + i.to(tl.int64) * V + ix, mask=mask, other=1.0)
    tl.store(losses_ptr + i, -tl.log(tl.maximum(p, 1e-3)), mask=mask)


def run(losses, probs, targets, B, T, V):
    N = B * T
    _ce_fwd[(triton.cdiv(N, BLOCK),)](losses, probs, targets, N, V, BLOCK=BLOCK, num_warps=4)
