# Example: flash-attention-minimal on Triton, WebGPU and Metal

[`flash.cu`](flash.cu) is an **unmodified** copy of
[tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
(Apache-2.0, see [LICENSE](LICENSE)), a well-known ~100-line CUDA FlashAttention forward pass.
It mixes the kernel with PyTorch extension host code (`#include <torch/types.h>`, a `forward()`
that allocates tensors and launches with dynamic shared memory); kernelport compiles only the
device code with NVRTC.

`.kernelport/` is the workspace: the analyzer's view of the kernel, the input spec and PyTorch
reference (frozen after the original passed it), the ports, attempts and results.

| | status | notes |
|---|---|---|
| reference vs original CUDA | PASS 6/6 | RTX 5060 Laptop (sm_120) |
| Triton | PASS 6/6, **23x** the original | 2.1 ms vs 49.3 ms at 8 batches x 12 heads x 384 tokens x d=64, same GPU |
| WebGPU (WGSL) | PASS 6/6 | RTX 5060 via Vulkan; CI also runs it on llvmpipe and Apple Metal |
| Metal (MLX) | checked in CI | Apple M1 runner, on every push |

The kernel takes `Q, K, V` of shape `[B, heads, N, d]` and leaves `O`, the running row sum `l`
and the row max `m` in three inout buffers (the host starts them at 0, 0 and -inf: the spec uses
`{"dist": "full", "value": "-inf"}` for `m`). The analyzer identified all three as inout from
the code. The reference is plain softmax attention in float64, plus the final `m` and `l`.

## What validation showed

- **Tolerance boundary.** kernelport's float32 tolerance (rtol 1.3e-6, the torch.testing default)
  is never loosened. The original uses `__expf` and sequential float32 sums, and its own rounding
  stays inside that tolerance up to N = 384 keys with unit-variance inputs. At N = 512, one `l`
  value in 4096 was off by a relative 1.86e-6. With Q and K of std 4 (attention logits of std 16),
  `__expf` drifted by about 1e-5. So the frozen spec covers N <= 384 and Q, K up to std 2 (logits
  up to std 4), which includes GPT-2's head geometry (12 heads, d = 64). It does not claim larger
  sizes.
- **N must be a multiple of 32.** The original loads tiles without bounds checks, so other
  lengths read and write past the buffers. That's not tested because it would corrupt GPU
  memory; the spec uses multiples of 32. The ports mask their loads and work for any N.

## The ports

- **Triton** (`ports/triton/port.py`): tiled `tl.dot` in IEEE fp32 (Triton's default TF32 would
  be off by ~1e-3). Its first version used the same online softmax as the original. That
  failed one `l` value in 36,864 (relative 1.75e-6, from rounding repeated each time the max
  grows), so the port now takes the row max in a first pass and sums `l` in fp64. It passed on
  attempt 2.
- **WebGPU** (`ports/webgpu/port.py`): one invocation per query row like the original, `vec4` dot
  products and block-wise sums to stay within fp32 tolerance without fp64. Porting it found a
  wgpu-native bug: a `var` declared inside a loop without an initializer is zeroed only once,
  not on every iteration as WGSL requires. kernelport's webgpu runner now refuses such
  declarations, because browsers follow the spec and the same shader would give other results.
- **Metal** (`ports/metal/port.py`): the same algorithm with `metal::precise::exp`.

```bash
cd examples/flash-attention-minimal
kernelport status                 # stored results
kernelport verify                 # re-verify every port this machine can run
kernelport validate forward_kernel  # re-run the original CUDA kernel (needs NVIDIA)
```
