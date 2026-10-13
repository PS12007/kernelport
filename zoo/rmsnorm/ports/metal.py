"""Metal (MLX) port of rmsnorm: one threadgroup per row; threadgroup tree reduction of x^2."""

SOURCE = """
    threadgroup float red[256];
    uint C = uint(cols[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint base = row * C;
    float sq = 0.0f;
    for (uint c = tid; c < C; c += 256) {
        float v = static_cast<float>(x[base + c]);
        sq += v * v;
    }
    red[tid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = metal::precise::rsqrt(red[0] / float(C) + eps[0]);
    for (uint c = tid; c < C; c += 256) {
        y[base + c] = static_cast<T>(static_cast<float>(x[base + c]) * inv_rms * static_cast<float>(weight[c]));
    }
"""
THREADGROUP = (256, 1, 1)


def grid(rows):
    return (rows * 256, 1, 1)
