"""Triton port of matmul_naive: C = A @ B, tiled with tl.dot, fp32 accumulation."""
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")  # exact fp32 products, no TF32
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(A, B, C, M, N, K):
    _k[(triton.cdiv(M, 32), triton.cdiv(N, 32))](A, B, C, M, N, K, BM=32, BN=32, BK=32)
