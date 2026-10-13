"""Triton port of reduce_sum: out[0] += sum(x), accumulated in float32.

Each program sums one BLOCK-sized chunk and atomically adds its partial into
out[0] (CUDA: block reduction in shared memory + atomicAdd). out is inout:
the atomics add to its initial value.
"""

import triton
import triton.language as tl


@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr, tl.sum(x, axis=0))


def run(x, out, n):
    BLOCK = 4096
    _reduce_sum_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
