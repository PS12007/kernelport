"""Reference for relu: y = max(x, 0)."""

import torch


def reference(x, n):
    return torch.clamp_min(x, 0.0)
