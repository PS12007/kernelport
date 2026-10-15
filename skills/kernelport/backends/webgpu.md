# Porting CUDA to WebGPU (WGSL)

A WebGPU port is a WGSL compute shader. The mapping from CUDA is close to
one-to-one: a CUDA thread block is a WGSL **workgroup**, `__shared__` is
`var<workgroup>`, `__syncthreads()` is `workgroupBarrier()`.

kernelport runs ports through **wgpu-native** (Vulkan / Metal / D3D12) with only
core WebGPU features enabled (plus the standard `shader-f16`), so WGSL that passes
here is WGSL a browser accepts. Timings are wall-clock including submission and are
**not comparable** to native kernel times.

## Port file format

```python
"""WebGPU (WGSL) port of relu: y[i] = max(x[i], 0)."""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    for (var i = gid.x; i < n; i += nwg.x * 256u) {    // grid-stride: see the 65535 limit
        let v = x[i];
        y[i] = select(T(0.0), v, v > T(0.0));
    }
}
"""


def grid(n):                         # launch.json names; arrays are NumPy (use .shape)
    return (min((n + 255) // 256, 65535),)   # workgroups per dimension
```

- The entry point is **`fn main`**. `grid(...)` returns 1 to 3 workgroup counts
  (CUDA: the grid); each must be **≤ 65535**.
- **Do not declare the bindings.** The runner appends them:
  - `alias T = f32;` (`f16` for float16 cases, with `enable f16;`), so write
    `array<T>` code once for every dtype;
  - one storage buffer per tensor param **named exactly like the launch.json param**,
    in launch.json order: `var<storage, read>` for `input`, `read_write` for
    `output`/`inout`. Element type: `T` for dtype "T"; else `f32`, `f16`, `i32`, `u32`;
  - **int64 arrives as `vec2<u32>`** (low word in `.x`); core WGSL has no 64-bit ints.
    Index arrays are small and non-negative: use `ids[t].x`;
  - `params`: a uniform struct with every scalar param (`params.n`, `params.eps`).
    int32 → `i32`, float32 → `f32`. Convert before comparing: `u32(params.n)`.
- To change a buffer's declared type (atomics), set
  `BINDINGS = {"out": "array<atomic<u32>>"}` in the port.
- Shader errors are reported with line numbers **inside the WGSL string**.
- Supported dtypes: float32, float16 (when the adapter has `shader-f16`). bfloat16
  cases are skipped and named in the result (no bf16 type in WGSL).

## CUDA → WGSL mapping

| CUDA | WGSL |
|---|---|
| `__global__ void k(...)` | `@compute @workgroup_size(256) fn main(...)` |
| `blockDim.x` (256) | the constant in `@workgroup_size` |
| `blockIdx.x` | `@builtin(workgroup_id) wid: vec3<u32>` → `wid.x` |
| `threadIdx.x` | `@builtin(local_invocation_id) lid` or `@builtin(local_invocation_index) tid` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `@builtin(global_invocation_id) gid` → `gid.x` |
| `gridDim.x` | `@builtin(num_workgroups) nwg` → `nwg.x` |
| `if (i < n)` | `if (i < u32(params.n)) { ... }` |
| `__shared__ float red[256];` | `var<workgroup> red: array<f32, 256>;` (module scope) |
| `__syncthreads()` | `workgroupBarrier()` (must be reached by the whole workgroup: not inside a divergent `if`) |
| `__shfl_xor_sync` warp reductions | tree reduction in `var<workgroup>` memory (subgroups are optional in WebGPU) |
| `atomicAdd(int*)` | `atomicAdd(&buf[i], v)` on `array<atomic<i32>>` / `atomic<u32>` |
| `atomicAdd(float*)` | **no float atomics in core WGSL**: compare-exchange loop on the bits (below) |
| `expf, logf, sqrtf, rsqrtf, fmaxf` | `exp, log, sqrt, inverseSqrt, max` |
| `-INFINITY` | no inf literals in WGSL: use `-3.4028234e38` (rounds to -FLT_MAX) as the max identity. Not `-3.4028235e38`: that decimal is above the largest f32, which browsers reject (the runner refuses it too) |
| `(float)x[i]` / `(T)v` | `f32(x[i])` / `T(v)` |
| `c ? a : b` | `select(b, a, c)` (note the argument order) |
| `size_t` / `long long` offsets | `u32` (fine below 4G elements) |
| template `<typename T>` | `alias T` provided by the runner |

## Patterns

**Element-wise, large n**: one invocation per element only works up to
65535 × workgroup_size elements. Use a grid-stride loop with the grid capped at
65535 (shown above), or a 2-D grid.

**One workgroup per row** (softmax, layernorm, rmsnorm, cumsum): `grid(rows) =
(rows,)`; invocations stride over the row (`for (var c = tid; c < cols; c += 256u)`),
then a workgroup tree reduction:

```wgsl
var<workgroup> red: array<f32, 256>;

fn block_sum(v: f32, tid: u32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();                // outside the if: every invocation reaches it
    }
    let r = red[0];
    workgroupBarrier();                    // before red is reused
    return r;
}
```

**Float atomicAdd** (reduce_sum): reduce per workgroup, then one invocation adds
the partial with a compare-exchange loop (`BINDINGS = {"out": "array<atomic<u32>>"}`):

```wgsl
var old = atomicLoad(&out[0]);
loop {
    let r = atomicCompareExchangeWeak(&out[0], old, bitcast<u32>(bitcast<f32>(old) + partial));
    if (r.exchanged) { break; }
    old = r.old_value;
}
```

**Tiles** (transpose, tiled matmul): `var<workgroup> tile: array<array<T, 33>, 32>;`,
a 2-D `@workgroup_size(32, 8)`, `workgroupBarrier()` between load and store, exactly
like the CUDA tile.

**Scans**: port the CUDA scan as is (Hillis-Steele in workgroup memory + a
`var<workgroup> carry: f32`), with barriers where CUDA has `__syncthreads`.

## Buffer size limits

One storage binding is limited by the adapter: browsers default to 128 MiB, software
Vulkan (llvmpipe) allows 128 MiB, discrete GPUs allow up to 2 GiB here. A case whose
tensor is larger than the adapter's limit is not run at all: the result names it as
"cases this device cannot run, not tested" with the sizes. That is a device limit, not
a bug in your port, so do not change the port for it. Say in your report which cases
were not tested and why.

## Precision notes

- Compute in f32 for f16 data (`f32(x[i])`, store with `T(v)`), like CUDA's float
  accumulators.
- WGSL's `exp`, `log` and `sqrt` precision is implementation-defined (exp: about
  3 + 2·|x| ulp). That is within tolerance when the argument is small, as in
  softmax's `exp(x - max)`. Nothing is loosened for it.
- There is no `fma` contraction control; plain `a * b + c` is fine.

## WebGPU-specific bugs checklist

- [ ] No binding declarations in the WGSL (the runner adds them); names match launch.json.
- [ ] `grid()` returns ≤ 65535 per dimension; large 1-D inputs use a grid-stride loop.
- [ ] Every `workgroupBarrier()` is in uniform control flow (not inside `if (tid < s)`).
- [ ] Scalars converted before use: `u32(params.n)`, `f32(params.cols)`.
- [ ] No plain `out[0] = out[0] + v` from several workgroups (a race): use atomics.
- [ ] int64 inputs read as `.x` of `vec2<u32>`.
- [ ] Workgroup arrays sized by the constant workgroup size, not by a runtime value.
