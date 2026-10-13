"""CPU (Numba) port of saxpy, in place: y[i] = a * x[i] + y[i]."""

from numba import njit, prange


@njit(parallel=True)
def _saxpy(n, a, x, y):
    for i in prange(n):
        y[i] = a * x[i] + y[i]  # a is a Python float -> computed in float64, rounded on store


def run(n, a, x, y):
    _saxpy(n, a, x, y)
