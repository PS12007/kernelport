"""CPU (Numba) port of reduce_sum: out[0] += sum(x[0:n]).

The CUDA kernel reduces per block and atomically adds partials into out[0];
here prange does a parallel reduction and the single add happens afterwards.
"""

from numba import njit, prange


@njit(parallel=True)
def _sum(x, n):
    acc = 0.0  # float64 accumulator (CUDA uses float; float64 is at least as accurate)
    for i in prange(n):
        acc += x[i]
    return acc


def run(x, out, n):
    out[0] += _sum(x, n)  # out is inout and starts at 0: add, do not overwrite
