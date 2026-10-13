"""CPU (Numba) port of matmul: C[M, N] = A[M, K] @ B[K, N], row-major.

One output row per prange iteration. The k loop sits outside the j loop so B
is read row by row (contiguous) and the inner loop vectorises; a float64 row
accumulator keeps the result at least as accurate as the kernel's float sum.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _matmul(A, B, C, M, N, K):
    for i in prange(M):
        acc = np.zeros(N, np.float64)
        for k in range(K):
            a = float(A[i, k])
            for j in range(N):
                acc[j] += a * B[k, j]
        for j in range(N):
            C[i, j] = acc[j]


def run(A, B, C, M, N, K):
    _matmul(A, B, C, M, N, K)
