"""Metal (MLX) port of layernorm: one threadgroup per row, two-pass mean / biased variance in float."""

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
    uint C = uint(cols[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint base = row * C;
    float s = 0.0f;
    for (uint c = tid; c < C; c += 256) s += static_cast<float>(x[base + c]);
    float mean = block_sum(red, s, tid) / float(C);
    float q = 0.0f;
    for (uint c = tid; c < C; c += 256) {
        float d = static_cast<float>(x[base + c]) - mean;
        q += d * d;
    }
    float rstd = metal::precise::rsqrt(block_sum(red, q, tid) / float(C) + eps[0]);
    for (uint c = tid; c < C; c += 256) {
        float v = (static_cast<float>(x[base + c]) - mean) * rstd;
        y[base + c] = static_cast<T>(v * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]));
    }
"""
THREADGROUP = (256, 1, 1)


def grid(rows):
    return (rows * 256, 1, 1)
