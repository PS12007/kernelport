"""Metal (MLX) port of row-wise softmax: one 256-thread threadgroup per row, float math."""

HEADER = """
inline float block_reduce(threadgroup float* red, float v, uint tid, bool is_max) {
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] = is_max ? metal::max(red[tid], red[tid + s]) : red[tid] + red[tid + s];
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
    float m = -INFINITY;
    for (uint c = tid; c < C; c += 256) m = metal::max(m, static_cast<float>(x[base + c]));
    m = block_reduce(red, m, tid, true);
    float s = 0.0f;
    for (uint c = tid; c < C; c += 256) s += metal::precise::exp(static_cast<float>(x[base + c]) - m);
    float inv = 1.0f / block_reduce(red, s, tid, false);
    for (uint c = tid; c < C; c += 256) {
        y[base + c] = static_cast<T>(metal::precise::exp(static_cast<float>(x[base + c]) - m) * inv);
    }
"""
THREADGROUP = (256, 1, 1)


def grid(rows):
    return (rows * 256, 1, 1)
