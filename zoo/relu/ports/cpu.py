"""CPU (Numba) port of relu: y[i] = max(x[i], 0)."""

from numba import njit, prange


@njit(parallel=True)
def _relu(x, y, n):
    for i in prange(n):
        v = x[i]
        y[i] = v if v > 0 else 0


def run(x, y, n):
    _relu(x, y, n)
