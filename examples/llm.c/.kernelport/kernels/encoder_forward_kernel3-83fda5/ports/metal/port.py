"""Metal (MLX) port of llm.c encoder_forward_kernel3: out[b, t, :] = wte[inp[b, t], :] + wpe[t, :].

One threadgroup per (b, t) row; float add, stored as bf16 (as the original).
"""

SOURCE = """
    uint bt = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint cols = uint(C[0]);
    uint t = bt % uint(T[0]);
    int64_t src = int64_t(inp[bt]) * int64_t(cols);
    for (uint c = tid; c < cols; c += 256) {
        float v = static_cast<float>(wte[src + c]) + static_cast<float>(wpe[t * cols + c]);
        out[bt * cols + c] = static_cast<bfloat16_t>(v);
    }
"""
THREADGROUP = (256, 1, 1)


def grid(B, T):
    return (B * T * 256, 1, 1)
