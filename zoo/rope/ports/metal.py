"""Metal (MLX) port of rope (interleaved pairs): one 128-thread threadgroup per token."""

SOURCE = """
    uint t = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint H = uint(heads[0]);
    uint D = uint(head_dim[0]);
    uint half_d = D / 2;
    int64_t p = positions[t];
    for (uint j = tid; j < H * half_d; j += 128) {
        uint h = j / half_d, i = j % half_d;
        uint base = (t * H + h) * D + 2 * i;
        float x0 = static_cast<float>(x[base]);
        float x1 = static_cast<float>(x[base + 1]);
        float c = cos_cache[p * half_d + i];
        float s = sin_cache[p * half_d + i];
        y[base] = static_cast<T>(x0 * c - x1 * s);
        y[base + 1] = static_cast<T>(x0 * s + x1 * c);
    }
"""
THREADGROUP = (128, 1, 1)


def grid(tokens):
    return (tokens * 128, 1, 1)
