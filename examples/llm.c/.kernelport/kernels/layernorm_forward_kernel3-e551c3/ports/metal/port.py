"""Metal (MLX) port of llm.c layernorm_forward_kernel3: one threadgroup per row, two passes in float."""

HEADER = """
inline float block_sum(threadgroup float* red, float v, uint tid) {
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
"""

SOURCE = """
    threadgroup float red[256];
    uint cols = uint(C[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint base = row * cols;
    float s = 0.0f;
    for (uint c = tid; c < cols; c += 256) s += inp[base + c];
    float m = block_sum(red, s, tid) / float(cols);
    float q = 0.0f;
    for (uint c = tid; c < cols; c += 256) {
        float d = inp[base + c] - m;
        q += d * d;
    }
    float r = metal::precise::rsqrt(block_sum(red, q, tid) / float(cols) + 1e-5f);
    if (tid == 0) {
        mean[row] = m;
        rstd[row] = r;
    }
    for (uint c = tid; c < cols; c += 256) out[base + c] = r * (inp[base + c] - m) * weight[c] + bias[c];
"""
THREADGROUP = (256, 1, 1)


def grid(N):
    return (N * 256, 1, 1)
