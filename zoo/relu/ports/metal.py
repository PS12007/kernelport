"""Metal (MLX) port of relu: y[i] = max(x[i], 0)."""

SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < uint(n[0])) {
        T v = x[i];
        y[i] = v > T(0) ? v : T(0);
    }
"""
THREADGROUP = (256, 1, 1)


def grid(n):
    return (n, 1, 1)
