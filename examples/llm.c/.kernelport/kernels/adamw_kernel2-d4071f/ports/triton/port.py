"""Triton port of adamw_kernel2: fused AdamW update (lerp-based moments, one lane per parameter)."""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TUNE = {"BLOCK": [512, 1024, 2048, 4096], "NUM_WARPS": [4, 8]}


@triton.jit
def _adamw(p_ptr, g_ptr, m_ptr, v_ptr, n, lr, beta1, beta2, c1, c2, eps, wd,
           BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # CUDA: blockIdx.x*blockDim.x+threadIdx.x
    mask = offs < n                                                      # CUDA: if (i >= n) return
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    m = tl.fma(beta1, m, tl.fma(-beta1, g, g))     # CUDA: m = lerp(grad, m, beta1)
    v = tl.fma(beta2, v, tl.fma(-beta2, g * g, g * g))  # CUDA: v = lerp(grad*grad, v, beta2)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)
    m_hat = m / c1
    v_hat = v / c2
    p = p - lr * (m_hat / (tl.sqrt_rn(v_hat) + eps) + wd * p)
    tl.store(p_ptr + offs, p, mask=mask)


def run(params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate,
        beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay):
    grid = (triton.cdiv(num_parameters, BLOCK),)
    _adamw[grid](params_memory, grads_memory, m_memory, v_memory, num_parameters,
                 learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                 BLOCK=BLOCK, num_warps=NUM_WARPS)
