"""Reference for transpose: out = inp^T."""


def reference(inp, rows, cols):
    return inp.t().contiguous()
