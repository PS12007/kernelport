"""Triton port of embedding_gather: out[t, :] = table[ids[t], :].

One program per token (CUDA: one block per token), copying the row in BLOCK
chunks. ids are int64, so the table offset ids[t] * dim is computed in int64.
"""

import triton
import triton.language as tl


@triton.jit
def _gather_kernel(ids_ptr, table_ptr, out_ptr, dim, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    row = tl.load(ids_ptr + t)
    src = table_ptr + row * dim
    dst = out_ptr + t * dim
    for start in range(0, dim, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < dim
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def run(ids, table, out, tokens, dim):
    BLOCK = min(triton.next_power_of_2(dim), 1024)
    _gather_kernel[(tokens,)](ids, table, out, dim, BLOCK=BLOCK)
