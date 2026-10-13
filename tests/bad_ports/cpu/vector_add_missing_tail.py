"""BAD cpu port: loop stops one short (off-by-one), the last element is never written."""

from numba import njit, prange


@njit(parallel=True)
def _k(a, b, out, n):
    for i in prange(n - 1):
        out[i] = a[i] + b[i]


def run(a, b, out, n):
    _k(a, b, out, n)
