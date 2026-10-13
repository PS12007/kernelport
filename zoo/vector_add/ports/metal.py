"""Metal (MLX) port of vector_add: out[i] = a[i] + b[i]."""

SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < uint(n[0])) {
        out[i] = a[i] + b[i];
    }
"""
THREADGROUP = (256, 1, 1)


def grid(n):
    return (n, 1, 1)
