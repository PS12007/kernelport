"""Reference for rope with interleaved pairs (x[..., 2i], x[..., 2i+1])."""

import torch


def reference(positions, x, cos_cache, sin_cache):
    c = cos_cache[positions][:, None, :]  # [tokens, 1, head_dim/2]
    s = sin_cache[positions][:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = x0 * c - x1 * s
    y[..., 1::2] = x0 * s + x1 * c
    return y
