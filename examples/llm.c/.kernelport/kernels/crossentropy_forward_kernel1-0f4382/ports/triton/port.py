"""Triton port of crossentropy_forward_kernel1: losses[b,t] = -log(probs[b,t,targets[b,t]])."""

import triton
import triton.language as tl

BLOCK = 256
NUM_WARPS = 4
TUNE = {"BLOCK": [128, 256, 512, 1024], "NUM_WARPS": [2, 4, 8]}


@triton.jit
def _ce_fwd(losses_ptr, probs_ptr, targets_ptr, N, V, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # CUDA: i = blockIdx.x*blockDim.x+threadIdx.x (i = b*T+t)
    mask = i < N                                          # CUDA: if (i < B*T)
    ix = tl.load(targets_ptr + i, mask=mask, other=0)     # CUDA: ix = targets[b*T+t]
    # CUDA: probs_bt = probs + b*T*V + t*V == probs + i*V (int64 to be safe)
    p = tl.load(probs_ptr + i.to(tl.int64) * V + ix, mask=mask, other=1.0)
    tl.store(losses_ptr + i, -tl.log(p), mask=mask)       # CUDA: -logf(probs_bt[ix])


def run(losses, probs, targets, B, T, V):
    N = B * T
    _ce_fwd[(triton.cdiv(N, BLOCK),)](losses, probs, targets, N, V, BLOCK=BLOCK, num_warps=NUM_WARPS)
