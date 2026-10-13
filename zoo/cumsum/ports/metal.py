"""Metal (MLX) port of cumsum: per-row inclusive scan, 256-wide chunks (Hillis-Steele) plus a carry."""

SOURCE = """
    threadgroup float buf[256];
    threadgroup float carry;
    uint C = uint(cols[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint base = row * C;
    if (tid == 0) carry = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint start = 0; start < C; start += 256) {
        uint c = start + tid;
        buf[tid] = c < C ? static_cast<float>(x[base + c]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < 256; off <<= 1) {
            float add = tid >= off ? buf[tid - off] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf[tid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (c < C) y[base + c] = static_cast<T>(carry + buf[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) carry += buf[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
THREADGROUP = (256, 1, 1)


def grid(rows):
    return (rows * 256, 1, 1)
