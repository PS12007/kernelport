"""CPU (Numba) port of cumsum: inclusive prefix sum along each row of [rows, cols].

The CUDA kernel scans BLOCK-sized chunks in shared memory and carries the
running total; on the CPU a sequential running sum per row is the same scan.
"""

from numba import njit, prange


@njit(parallel=True)
def _cumsum(x, y, rows, cols):
    for r in prange(rows):
        acc = 0.0
        for c in range(cols):
            acc += x[r, c]
            y[r, c] = acc


def run(x, y, rows, cols):
    _cumsum(x, y, rows, cols)
