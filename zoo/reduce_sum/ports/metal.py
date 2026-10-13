"""Metal (MLX) port of reduce_sum: out += sum(x).

Each threadgroup reduces a strided slice in threadgroup memory and atomically adds its
partial into out_out (outputs are atomic, initialised to 0). out is inout: one thread
also adds its initial value.
"""

ATOMIC_OUTPUTS = True
INIT_VALUE = 0.0

SOURCE = """
    threadgroup float red[256];
    uint n_ = uint(n[0]);
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint groups = threadgroups_per_grid.x;
    float acc = 0.0f;
    for (uint i = group * 256 + tid; i < n_; i += groups * 256) acc += static_cast<float>(x[i]);
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float partial = red[0] + (group == 0 ? out[0] : 0.0f);   // inout: add the initial value once
        atomic_fetch_add_explicit(&out_out[0], partial, memory_order_relaxed);
    }
"""
THREADGROUP = (256, 1, 1)


def grid(n):
    groups = min((n + 255) // 256, 1024)
    return (groups * 256, 1, 1)
