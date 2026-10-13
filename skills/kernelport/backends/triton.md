# Porting CUDA to Triton

Triton programs operate on **blocks of values**, not single threads. One Triton
*program* (`tl.program_id`) does the work of one CUDA *thread block*; the block's
threads become the lanes of `tl.arange` vectors. You never write `__syncthreads`
or shared memory by hand: block-wide operations (`tl.sum`, `tl.max`, `tl.cumsum`)
do it for you.

Verified on NVIDIA (triton / triton-windows) and AMD ROCm (same port file).

## Port file format

```python
"""Triton port of <kernel>: <what it computes>."""

import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # CUDA: blockIdx.x * blockDim.x + threadIdx.x
    mask = offs < n                                          # CUDA: if (i < n)
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)


def run(x, y, n):
    BLOCK = 1024
    _kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
```

- `run(...)` takes the **launch.json param names** (any subset). Tensors are GPU
  torch tensors; outputs are pre-filled with NaN and must be written **in place**.
  Scalars are plain Python numbers. Return value is ignored.
- Choose your own grid; it does not have to match the CUDA grid.
- The output dtype is fixed by the tensor you store into: `tl.store` casts a float32
  value to the pointer's element type (fp16/bf16) automatically.

## CUDA → Triton mapping

| CUDA | Triton |
|---|---|
| `blockIdx.x`, `gridDim.x` | `tl.program_id(0)`, `tl.num_programs(0)` |
| `threadIdx.x` over `blockDim.x` | `tl.arange(0, BLOCK)` (BLOCK must be a power of 2 `tl.constexpr`) |
| `if (i < n)` bounds check | `mask = offs < n`, pass `mask=` to every `tl.load`/`tl.store` |
| 2D tile `(ty, tx)` | `rows = r0 + tl.arange(0, BM)[:, None]`, `cols = c0 + tl.arange(0, BN)[None, :]`, `ptr + rows * stride + cols` |
| grid-stride / chunk loop over a row | `for start in range(0, cols, BLOCK): offs = start + tl.arange(0, BLOCK)` |
| `__shared__` buffers + `__syncthreads` | keep values in block tensors; no sync needed |
| block reduction (shared-memory tree) | `tl.sum(x, axis=0)`, `tl.max(x, axis=0)`, `tl.min`, `tl.reduce(x, 0, combine_fn)` |
| warp shuffles `__shfl_xor_sync` reduce | same as block reduction: `tl.sum` / `tl.max` over the block tensor |
| prefix sum / scan | `tl.cumsum(x, axis=0)` or `tl.associative_scan(x, 0, combine_fn)`; carry the running total across chunks yourself |
| `atomicAdd(&p[i], v)` | `tl.atomic_add(p_ptr + i, v, mask=m)`; also `tl.atomic_max/min/xchg/cas` |
| `float4` vector loads | not needed; contiguous `tl.arange` loads are vectorised automatically |
| `(float)h` for `__half`/bf16 | `x.to(tl.float32)` |
| `(T)acc` store | just `tl.store(ptr, acc)`; it is cast to the pointer dtype |
| `expf`, `rsqrtf`, `sqrtf`, `fmaxf` | `tl.exp`, `tl.rsqrt`, `tl.sqrt_rn` (exact) / `tl.sqrt` (approx), `tl.maximum` |
| `-INFINITY` | `float("-inf")` |
| `size_t` index math | cast before multiplying: `row = tl.program_id(0).to(tl.int64)` |
| gather `table[ids[t] * dim + d]` | `idx = tl.load(ids_ptr + t)`; `tl.load(table_ptr + idx * dim + offs, mask=...)` |
| matmul inner loop | `acc += tl.dot(a, b, input_precision="ieee")` (see TF32 warning) |
| `template <typename T>` | nothing to do: Triton specialises on the pointer dtype |
| `#define BLOCK 256` | `BLOCK: tl.constexpr` argument |

## Patterns

**Row-wise op on a [rows, cols] matrix (softmax, layernorm, rmsnorm):**
one program per row. If `cols` may exceed one block, loop over the row in chunks;
for softmax use an online max/sum (rescale the running sum when the max grows),
or do one pass for the max and a second for the sum.

```python
row = tl.program_id(0).to(tl.int64)
acc = tl.zeros((BLOCK,), tl.float32)
for start in range(0, cols, BLOCK):
    offs = start + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * cols + offs, mask=offs < cols, other=0.0).to(tl.float32)
    acc += x * x
mean_sq = tl.sum(acc, axis=0) / cols
```

**Masked loads feeding a reduction:** choose `other=` so masked lanes are neutral:
`0.0` for sums, `float("-inf")` for max, `float("inf")` for min.

**Full reduction to one value:** either `tl.atomic_add` each program's partial into
the output (the output must start at the value the CUDA kernel expects; inout
params arrive with their initial values), or one program looping over everything.

**Scan across chunks:** `tl.cumsum` within the chunk, add the carry, then
`carry += tl.sum(chunk)`.

**Matmul:** tile with `tl.dot` on `[BM, BK] x [BK, BN]` blocks, accumulate in a
float32 `tl.zeros((BM, BN), tl.float32)`, mask the K tail with `other=0.0`.
`tl.dot` needs every block dim ≥ 16.

## Precision notes

- **TF32:** `tl.dot` on float32 inputs uses TF32 by default (~1e-3 relative error).
  That FAILS verification. Always pass `input_precision="ieee"` for float32.
- Accumulate in float32 (`.to(tl.float32)`) like the CUDA kernel does; accumulating
  in fp16/bf16 will fail.
- `tl.sqrt` and `/` use fast approximate instructions (≤ 2 ulp); that is within
  tolerance. If a result must be correctly rounded, use `tl.sqrt_rn` / `tl.div_rn`.

## Triton-specific bugs checklist

- [ ] Every `tl.load`/`tl.store` that can go out of bounds has a `mask`.
- [ ] `tl.arange` size is a power of two `constexpr`; use `triton.next_power_of_2(n)`
      and mask the excess, or loop in chunks.
- [ ] Masked lanes use a neutral `other=` value for the reduction they feed.
- [ ] Large offsets cast to int64 before multiplying (rows × cols > 2^31).
- [ ] `run()` writes outputs in place; it does not allocate and return new tensors.
- [ ] `run()` parameter names match launch.json exactly.
- [ ] float32 `tl.dot` uses `input_precision="ieee"`.
- [ ] Integer outputs store integer values (cast with `.to(tl.int32)` etc. if needed).
- [ ] No Python-side work on tensor values inside `run()` (no `.item()`, no torch
      math replacing the kernel); the port must compute on the GPU in Triton.
