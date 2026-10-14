"""Triton port of crossentropy_softmax_backward_kernel1:
dlogits[b,t,v] += (probs[b,t,v] - (v == targets[b,t])) * dlosses[b,t]."""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TUNE = {"BLOCK": [512, 1024, 2048, 4096], "NUM_WARPS": [4, 8]}


@triton.jit
def _ce_sm_bwd(dlogits_ptr, dlosses_ptr, probs_ptr, targets_ptr, V, BLOCK: tl.constexpr):
    # CUDA uses one thread per (b,t,v); here program axis 0 = row b*T+t, axis 1 = chunk of v
    row = tl.program_id(0).to(tl.int64)
    v = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)       # CUDA: v = i % V
    mask = v < V                                            # CUDA: if (i < B*T*V)
    dloss = tl.load(dlosses_ptr + row)                      # CUDA: dlosses[b*T+t]
    ix = tl.load(targets_ptr + row)                         # CUDA: targets[b*T+t]
    base = row * V                                          # CUDA: b*T*V + t*V
    p = tl.load(probs_ptr + base + v, mask=mask, other=0.0)
    indicator = tl.where(v == ix, 1.0, 0.0)                 # CUDA: v == ix ? 1.0f : 0.0f
    d = tl.load(dlogits_ptr + base + v, mask=mask, other=0.0)
    tl.store(dlogits_ptr + base + v, d + (p - indicator) * dloss, mask=mask)  # CUDA: +=


def run(dlogits, dlosses, probs, targets, B, T, V):
    grid = (B * T, triton.cdiv(V, BLOCK))
    _ce_sm_bwd[grid](dlogits, dlosses, probs, targets, V, BLOCK=BLOCK, num_warps=NUM_WARPS)
