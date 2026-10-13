"""Reference for cumsum: inclusive prefix sum along each row."""

import torch


def reference(x):
    return torch.cumsum(x, dim=-1)
