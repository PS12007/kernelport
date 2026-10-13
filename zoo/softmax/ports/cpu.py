"""CPU (Numba) port of row-wise softmax over a [rows, cols] matrix.

One row per prange iteration (CUDA: one block per row). Max-subtraction for
stability, float64 math, one exp per element.
"""

import math

from numba import njit, prange


@njit(parallel=True)
def _softmax(x, y, rows, cols):
    for r in prange(rows):
        m = -math.inf
        for c in range(cols):
            m = max(m, float(x[r, c]))
        s = 0.0
        for c in range(cols):  # store exp(x - max) once; rescale in place below
            e = math.exp(float(x[r, c]) - m)
            y[r, c] = e
            s += e
        inv = 1.0 / s
        for c in range(cols):
            y[r, c] = y[r, c] * inv


def run(x, y, rows, cols):
    _softmax(x, y, rows, cols)
