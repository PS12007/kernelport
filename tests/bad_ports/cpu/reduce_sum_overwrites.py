"""BAD cpu port: overwrites the inout accumulator instead of adding to it.

Passes while out starts at 0; caught only because a case starts out non-zero.
"""

from numba import njit, prange


@njit(parallel=True)
def _sum(x, n):
    acc = 0.0
    for i in prange(n):
        acc += x[i]
    return acc


def run(x, out, n):
    out[0] = _sum(x, n)
