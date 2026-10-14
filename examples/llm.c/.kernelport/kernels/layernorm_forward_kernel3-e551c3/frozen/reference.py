import torch


def reference(inp, weight, bias, N, C):
    m = inp.mean(-1)
    xc = inp - m[:, None]
    s = torch.rsqrt((xc * xc).mean(-1) + 1e-5)
    return {"out": xc * s[:, None] * weight + bias, "mean": m, "rstd": s}
