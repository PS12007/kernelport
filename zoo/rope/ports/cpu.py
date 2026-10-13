"""CPU (Numba) port of rope with interleaved pairs.

For token t at position p = positions[t], head h, pair i:
  y[t, h, 2i]   = x0 * cos[p, i] - x1 * sin[p, i]
  y[t, h, 2i+1] = x0 * sin[p, i] + x1 * cos[p, i]
"""

from numba import njit, prange


@njit(parallel=True)
def _rope(positions, x, cos_cache, sin_cache, y, tokens, heads, head_dim):
    half = head_dim // 2
    for t in prange(tokens):
        p = positions[t]
        for h in range(heads):
            for i in range(half):
                x0 = float(x[t, h, 2 * i])
                x1 = float(x[t, h, 2 * i + 1])
                c = float(cos_cache[p, i])
                s = float(sin_cache[p, i])
                y[t, h, 2 * i] = x0 * c - x1 * s
                y[t, h, 2 * i + 1] = x0 * s + x1 * c


def run(positions, x, cos_cache, sin_cache, y, tokens, heads, head_dim):
    _rope(positions, x, cos_cache, sin_cache, y, tokens, heads, head_dim)
