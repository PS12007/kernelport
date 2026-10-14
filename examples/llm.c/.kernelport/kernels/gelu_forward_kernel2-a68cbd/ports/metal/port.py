"""Metal (MLX) port of llm.c gelu_forward_kernel2: bf16 in/out, float math, tanh approximation."""

SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i < uint(N[0])) {
        float x = static_cast<float>(inp[i]);
        float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        out[i] = static_cast<bfloat16_t>(0.5f * x * (1.0f + metal::precise::tanh(z)));
    }
"""
THREADGROUP = (256, 1, 1)


def grid(N):
    return (N, 1, 1)
