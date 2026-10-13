"""BUG: uses an undefined Triton function, so compilation fails."""
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.not_a_function(a_ptr + offs))


def run(a, b, out, n):
    _k[(triton.cdiv(n, 256),)](a, b, out, n, BLOCK=256)
