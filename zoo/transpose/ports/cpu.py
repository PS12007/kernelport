"""CPU (Numba) port of transpose: out[c, r] = inp[r, c] for a [rows, cols] input.

The CUDA kernel stages 32x32 tiles in shared memory; on the CPU the same tiling
keeps both the reads and the writes inside cache lines.
"""

from numba import njit, prange

TILE = 32


@njit(parallel=True)
def _transpose(inp, out, rows, cols):
    n_col_tiles = (cols + TILE - 1) // TILE
    for ct in prange(n_col_tiles):
        c0 = ct * TILE
        c1 = min(c0 + TILE, cols)
        for r0 in range(0, rows, TILE):
            r1 = min(r0 + TILE, rows)
            for c in range(c0, c1):
                for r in range(r0, r1):
                    out[c, r] = inp[r, c]


def run(inp, out, rows, cols):
    _transpose(inp, out, rows, cols)
