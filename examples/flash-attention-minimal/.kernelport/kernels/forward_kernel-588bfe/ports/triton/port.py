"""Triton port of flash-attention-minimal's forward_kernel: tiled attention, softmax in two passes.

One program per (batch*head, block of BM queries). Pass 1 streams the key blocks for the row
max m; pass 2 streams them again for p = exp(s - m), the row sum l and O = (p V) / l. The CUDA
kernel instead rescales running sums every time the max grows (online softmax); doing it in two
passes avoids that repeated rounding, and l is accumulated in fp64, so the result stays within
fp32 tolerance at sizes where single-pass fp32 rounding does not. Unlike the original, N need not
be a multiple of 32 (masked loads). tl.dot runs in IEEE fp32 (Triton's default is TF32, ~1e-3
error). Writes the final O, l and m, which the original leaves in its inout buffers.
"""

import triton
import triton.language as tl


@triton.jit
def _flash_fwd(Q, K, V, O, L, M, N, scale, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    cols = tl.arange(0, D)
    base = bh * N * D
    row_ok = rows < N
    q = tl.load(Q + base + rows[:, None] * D + cols[None, :], mask=row_ok[:, None], other=0.0)

    m_i = tl.full((BM,), float("-inf"), tl.float32)
    for start in range(0, N, BN):                      # pass 1: row max of the scores
        keys = start + tl.arange(0, BN)
        k = tl.load(K + base + keys[:, None] * D + cols[None, :], mask=(keys < N)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        m_i = tl.maximum(m_i, tl.max(tl.where((keys < N)[None, :], s, float("-inf")), axis=1))

    l_i = tl.zeros((BM,), tl.float64)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(0, N, BN):                      # pass 2: exp, row sum, P @ V
        keys = start + tl.arange(0, BN)
        key_ok = keys < N
        k = tl.load(K + base + keys[:, None] * D + cols[None, :], mask=key_ok[:, None], other=0.0)
        v = tl.load(V + base + keys[:, None] * D + cols[None, :], mask=key_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        p = tl.where(key_ok[None, :], tl.exp(s - m_i[:, None]), 0.0)
        l_i += tl.sum(p, axis=1).to(tl.float64)
        acc += tl.dot(p, v, input_precision="ieee")

    l32 = l_i.to(tl.float32)
    tl.store(O + base + rows[:, None] * D + cols[None, :], acc / l32[:, None], mask=row_ok[:, None])
    tl.store(L + bh * N + rows, l32, mask=row_ok)
    tl.store(M + bh * N + rows, m_i, mask=row_ok)


def run(Q, K, V, N, d, softmax_scale, l, m, O):
    B, nh = Q.shape[0], Q.shape[1]
    BM = BN = 32
    grid = (B * nh, triton.cdiv(N, BM))
    _flash_fwd[grid](Q, K, V, O, l, m, N, softmax_scale, D=d, BM=BM, BN=BN)
