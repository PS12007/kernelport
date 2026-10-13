"""BAD cpu port: exclusive scan (writes before adding) instead of inclusive."""

from numba import njit, prange


@njit(parallel=True)
def _k(x, y, rows, cols):
    for r in prange(rows):
        acc = 0.0
        for c in range(cols):
            y[r, c] = acc
            acc += x[r, c]


def run(x, y, rows, cols):
    _k(x, y, rows, cols)
