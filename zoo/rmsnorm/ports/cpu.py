"""CPU (Numba) port of rmsnorm over the last dim: y = x / sqrt(mean(x^2) + eps) * weight."""

import math

from numba import njit, prange


@njit(parallel=True)
def _rmsnorm(x, weight, y, rows, cols, eps):
    for r in prange(rows):
        sq = 0.0
        for c in range(cols):
            v = float(x[r, c])
            sq += v * v
        inv_rms = 1.0 / math.sqrt(sq / cols + eps)
        for c in range(cols):
            y[r, c] = x[r, c] * inv_rms * weight[c]


def run(x, weight, y, rows, cols, eps):
    _rmsnorm(x, weight, y, rows, cols, eps)
