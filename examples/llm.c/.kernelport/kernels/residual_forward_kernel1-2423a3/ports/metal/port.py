"""Metal (MLX) port of llm.c residual_forward_kernel1: out = bf16(float(a) + float(b))."""

SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < uint(N[0])) {
        out[i] = static_cast<bfloat16_t>(static_cast<float>(inp1[i]) + static_cast<float>(inp2[i]));
    }
"""
THREADGROUP = (256, 1, 1)


def grid(N):
    return (N, 1, 1)
