"""Reference for rmsnorm over the last dimension."""

import torch


def reference(x, weight, eps):
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms * weight
