"""Reference for softmax: softmax over the last dimension (cols)."""

import torch


def reference(x, rows, cols):
    return torch.softmax(x, dim=-1)
