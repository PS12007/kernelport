"""CPU (Numba) port of layernorm over the last dim of a [rows, cols] matrix.

y = (x - mean) / sqrt(var + eps) * gamma + beta, two-pass (mean, then variance
of x - mean) like the CUDA kernel, in float64.
"""

import math

from numba import njit, prange


@njit(parallel=True)
def _layernorm(x, gamma, beta, y, rows, cols, eps):
    for r in prange(rows):
        s = 0.0
        for c in range(cols):
            s += x[r, c]
        mean = s / cols
        sq = 0.0
        for c in range(cols):
            d = x[r, c] - mean
            sq += d * d
        rstd = 1.0 / math.sqrt(sq / cols + eps)  # biased variance, as in the kernel
        for c in range(cols):
            y[r, c] = (x[r, c] - mean) * rstd * gamma[c] + beta[c]


def run(x, gamma, beta, y, rows, cols, eps):
    _layernorm(x, gamma, beta, y, rows, cols, eps)
