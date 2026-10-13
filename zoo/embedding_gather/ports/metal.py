"""Metal (MLX) port of embedding_gather: out[t, :] = table[ids[t], :]; one threadgroup per token."""

SOURCE = """
    uint t = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint D = uint(dim[0]);
    int64_t src = ids[t] * int64_t(D);
    uint dst = t * D;
    for (uint d = tid; d < D; d += 128) out[dst + d] = table[src + d];
"""
THREADGROUP = (128, 1, 1)


def grid(tokens):
    return (tokens * 128, 1, 1)
