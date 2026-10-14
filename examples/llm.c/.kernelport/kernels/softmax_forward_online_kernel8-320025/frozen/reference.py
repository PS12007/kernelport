import torch


def reference(inp, N, C):
    return torch.softmax(inp, dim=-1)
