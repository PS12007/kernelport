"""Reference for layernorm over the last dimension."""

import torch


def reference(x, gamma, beta, cols, eps):
    return torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps)
