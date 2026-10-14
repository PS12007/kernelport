"""Metal (MLX) port of llm.c softmax_forward_online_kernel8: one threadgroup per row.

Each thread keeps an online (max, sum) pair over its strided columns; pairs are merged in
threadgroup memory. Any C >= 1 works (the original needs C >= blockDim.x).
"""

HEADER = """
inline void merge(thread float& m, thread float& s, float m2, float s2) {
    float mx = metal::max(m, m2);
    float a = (m == -INFINITY) ? 0.0f : s * metal::precise::exp(m - mx);
    float b = (m2 == -INFINITY) ? 0.0f : s2 * metal::precise::exp(m2 - mx);
    m = mx;
    s = a + b;
}
"""

SOURCE = """
    threadgroup float red_m[256];
    threadgroup float red_s[256];
    uint cols = uint(C[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint base = row * cols;
    float m = -INFINITY, s = 0.0f;
    for (uint c = tid; c < cols; c += 256) merge(m, s, inp[base + c], 1.0f);
    red_m[tid] = m;
    red_s[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 128; k > 0; k >>= 1) {
        if (tid < k) {
            float mm = red_m[tid], ss = red_s[tid];
            merge(mm, ss, red_m[tid + k], red_s[tid + k]);
            red_m[tid] = mm;
            red_s[tid] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red_m[0], row_sum = red_s[0];
    for (uint c = tid; c < cols; c += 256) out[base + c] = metal::precise::exp(inp[base + c] - row_max) / row_sum;
"""
THREADGROUP = (256, 1, 1)


def grid(N):
    return (N * 256, 1, 1)
