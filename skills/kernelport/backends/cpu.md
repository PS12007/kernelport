# Porting CUDA to the CPU (Numba)

A cpu port is plain Python whose hot loops are compiled by Numba (`@njit`).
`prange` loops are split across CPU threads, so one `prange` iteration plays the
role of one CUDA thread block (or one thread, for element-wise kernels). There is
no shared memory, no `__syncthreads`, no warps: code inside one iteration runs
sequentially, which makes most block-level CUDA tricks unnecessary.

Runs on any machine with Numba (verified so far on Windows x86-64).
Timings are wall-clock and **not comparable** to GPU kernel times; the report says so.

## Port file format

```python
"""CPU (Numba) port of <kernel>: <what it computes>."""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _softmax(x, y, rows, cols):
    for r in prange(rows):            # CUDA: one block per row (blockIdx.x)
        m = -math.inf
        for c in range(cols):         # CUDA: threads striding over c, then a block reduction
            m = max(m, float(x[r, c]))
        s = 0.0
        for c in range(cols):
            s += math.exp(float(x[r, c]) - m)
        for c in range(cols):
            y[r, c] = math.exp(float(x[r, c]) - m) / s


def run(x, y, rows, cols):
    _softmax(x, y, rows, cols)
```

- `run(...)` takes **the launch.json parameter names** (any subset, any order).
- Tensors arrive as **C-contiguous NumPy arrays with the launch.json shapes**
  (`["rows", "cols"]` gives a 2-D array; index it as `x[r, c]`, or `x.ravel()` for
  the CUDA flat index). Integer params (`int64` ids, positions) keep their dtype.
- **Write outputs in place** (`y[i] = ...`, `y[:] = ...`). Outputs are pre-filled
  with NaN, so an element you never write shows up as "unexpected NaN".
  `y = np.maximum(x, 0)` only rebinds the local name, and **returning an array is
  rejected** with an error.
- `inout` params hold their initial values: add to them, do not overwrite
  (`out[0] += total`).
- `run` itself is ordinary Python and may call several `@njit` functions. The
  kernel's work belongs in `@njit` loops; use NumPy in `run` only for glue
  (views, `ravel`, zero-filling scratch), never `np.cumsum`/`np.matmul`-style
  calls that replace the kernel.
- Supported dtypes: **float32, float64**. float16/bfloat16 cases are skipped and
  named in the result (NumPy has no bfloat16; Numba does not compile float16 math).

## CUDA → Numba mapping

| CUDA | Numba (CPU) |
|---|---|
| `__global__ void k(...)` + `<<<grid, block>>>` | `@njit(parallel=True) def _k(...)` called from `run` |
| `i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n)` | `for i in prange(n):` (no guard needed, the range is exact) |
| one block per row (`blockIdx.x` = row) | `for r in prange(rows):` with sequential inner loops |
| 2-D grid (`blockIdx.y`, `blockIdx.x`) | `prange` over the outer index, `range` over the inner one |
| threads striding over a row + shared-memory reduction | a plain `for c in range(cols): acc += ...` inside the iteration |
| `__shfl_xor_sync` / warp reductions | same: a sequential loop; no warps exist |
| `__syncthreads()` between phases | nothing: phases inside one iteration already run in order |
| `__shared__ float tile[32][33]` | not needed; for cache-friendly access loop over tiles (see transpose) |
| per-block scratch (`__shared__` buffer) | `buf = np.empty(n)` inside the `prange` body (one per iteration) |
| `atomicAdd(out, partial)` from every block | prange **scalar** reduction, then one write: `acc += v` in the loop, `out[0] += acc` after |
| `atomicAdd(&hist[bin], 1)` (scatter) | **no atomics**: serial loop, or per-thread private histograms merged after |
| `expf`, `logf`, `sqrtf`, `rsqrtf(x)` | `math.exp`, `math.log`, `math.sqrt`, `1.0 / math.sqrt(x)` |
| `fmaxf(a, b)` / `fminf` | `max(a, b)` / `min(a, b)` |
| `(float)x[i]` on half inputs | `float(x[i])` (computes in float64) |
| `long long` index / `size_t` offsets | Python ints in Numba are int64; no overflow at these sizes |
| `__launch_bounds__`, block size, `#pragma unroll` | ignore; Numba/LLVM vectorises inner loops itself |

## Patterns

**Element-wise** (vector_add, relu, saxpy): one `prange` over `n`.

**Row-wise reductions** (softmax, layernorm, rmsnorm): `prange` over rows; each
iteration does the kernel's phases (max, sum, normalise) as sequential loops over
the row. Keep the kernel's math: two-pass mean/variance stays two-pass, biased
variance stays biased (`/ cols`, not `/ (cols - 1)`).

**Whole-array reduction** (reduce_sum): scalar reduction across a `prange`:

```python
@njit(parallel=True)
def _sum(x, n):
    acc = 0.0
    for i in prange(n):
        acc += x[i]          # Numba recognises scalar +=, *=, and m = max(m, ...) as reductions
    return acc

def run(x, out, n):
    out[0] += _sum(x, n)     # out is inout: add
```

**Scans** (cumsum): `prange` over rows, a running sum along the row. A
Hillis-Steele or Blelloch scan is a GPU technique; on the CPU the sequential scan
is both simpler and faster. Inclusive vs exclusive must match the kernel.

**Matmul**: `prange` over output rows; put `k` outside `j` so `B[k, :]` is read
contiguously and the `j` loop vectorises; accumulate a row in a float64 buffer.

**Transpose / strided access**: tile both loops (e.g. 32×32) so reads and writes
stay in cache. Correctness does not depend on it, only speed.

**Gathers** (embedding, rope): `prange` over tokens; index with the int64 ids
directly (`table[ids[t], d]`).

## Precision notes

- Float literals are float64: `0.5 * x[i]` with a float32 `x` computes in float64.
  `float32 * float32` stays float32; `math.exp` of a float32 returns float32.
- Computing in float64 (`float(x[i])`, `acc = 0.0`) is **always allowed and
  recommended**: it costs little on a CPU and is at least as accurate as the
  kernel's float math. The result is rounded to float32 on store.
- Leave `fastmath` off (the default). `fastmath=True` lets LLVM reorder sums and
  assume no NaN/inf; the tolerances are not loosened for it.
- Scalar Python args (`a`, `eps`) arrive as Python floats, i.e. float64.

## CPU-specific bugs checklist

- [ ] Every output element written in place; nothing returned from `run`.
- [ ] `inout` params added to (or updated), not overwritten.
- [ ] No writes to the **same array element from different `prange` iterations**
      (`hist[0] += 1` or `out[0] += v` inside `prange` is a data race; Numba has no
      atomics on the CPU and silently loses updates). Reduce into a scalar instead.
- [ ] Inner loops use `range`, not `prange` (nested `prange` runs serially anyway,
      and races are easy to introduce).
- [ ] Loop bounds match the kernel's guards (`range(n)`, not `range(n - 1)`).
- [ ] No `import` inside an `@njit` function (unsupported); import at module top.
- [ ] Only NumPy/`math` calls Numba supports inside `@njit` (no torch, no Python
      objects, no `print` of arrays in hot loops).
- [ ] No `cache=True`: it writes cache files next to the port inside `.kernelport/`.
