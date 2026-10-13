"""CPU (Numba) port of embedding_gather: out[t, :] = table[ids[t], :]."""

from numba import njit, prange


@njit(parallel=True)
def _gather(ids, table, out, tokens, dim):
    for t in prange(tokens):
        row = ids[t]  # int64, like the kernel's long long
        for d in range(dim):
            out[t, d] = table[row, d]


def run(ids, table, out, tokens, dim):
    _gather(ids, table, out, tokens, dim)
