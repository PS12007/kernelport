"""Reference for reduce_sum: out = out + sum(x) (out starts at zero)."""


def reference(x, out, n):
    return {"out": out + x.sum()}
