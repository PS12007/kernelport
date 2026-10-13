"""BAD cpu port: unbiased variance (divides by cols - 1) instead of the kernel's biased variance."""

import math

from numba import njit, prange


@njit(parallel=True)
def _k(x, gamma, beta, y, rows, cols, eps):
    for r in prange(rows):
        s = 0.0
        for c in range(cols):
            s += x[r, c]
        mean = s / cols
        sq = 0.0
        for c in range(cols):
            d = x[r, c] - mean
            sq += d * d
        rstd = 1.0 / math.sqrt(sq / max(cols - 1, 1) + eps)
        for c in range(cols):
            y[r, c] = (x[r, c] - mean) * rstd * gamma[c] + beta[c]


def run(x, gamma, beta, y, rows, cols, eps):
    _k(x, gamma, beta, y, rows, cols, eps)
