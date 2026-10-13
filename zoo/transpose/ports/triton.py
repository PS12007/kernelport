"""Triton port of transpose: out[c, r] = inp[r, c] for a [rows, cols] input.

One program per 32x32 tile (CUDA: one block per tile; the shared-memory staging
is unnecessary because Triton loads the whole tile into registers).
"""

import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(in_ptr, out_ptr, rows, cols, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    r = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    c = tl.program_id(1) * TILE_C + tl.arange(0, TILE_C)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    tile = tl.load(in_ptr + r[:, None].to(tl.int64) * cols + c[None, :], mask=mask)
    tl.store(out_ptr + c[None, :].to(tl.int64) * rows + r[:, None], tile, mask=mask)


def run(inp, out, rows, cols):
    TILE = 32
    grid = (triton.cdiv(rows, TILE), triton.cdiv(cols, TILE))
    _transpose_kernel[grid](inp, out, rows, cols, TILE_R=TILE, TILE_C=TILE)
