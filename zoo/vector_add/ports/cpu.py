"""CPU (Numba) port of vector_add: out[i] = a[i] + b[i]."""

from numba import njit, prange


@njit(parallel=True)
def _vector_add(a, b, out, n):
    for i in prange(n):  # CUDA: one thread per i, guarded by i < n
        out[i] = a[i] + b[i]


def run(a, b, out, n):
    _vector_add(a, b, out, n)
