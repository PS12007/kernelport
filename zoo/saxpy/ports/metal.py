"""Metal (MLX) port of saxpy: y = a * x + y. y is inout: read `y`, write `y_out`."""

SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < uint(n[0])) {
        y_out[i] = static_cast<T>(a[0] * static_cast<float>(x[i]) + static_cast<float>(y[i]));
    }
"""
THREADGROUP = (256, 1, 1)


def grid(n):
    return (n, 1, 1)
