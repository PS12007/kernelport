"""Metal (MLX) port of flash-attention-minimal's forward_kernel: one thread per query row.

Same algorithm as the WebGPU port: softmax in two passes (row max m, then p = exp(s - m), the
row sum l and O = (p V) / l), float4 dot products for the scores and sums per block of 32 keys,
which keeps fp32 rounding small. metal::precise::exp, since Metal compiles with fast-math. Head
dims d <= 128 that are multiples of 4; N need not be a multiple of 32. The inouts are written as
l_out, m_out and O_out (their final values, as the original leaves them).
"""

HEADER = """
inline float score(thread const float4* q, const device float* K, uint koff, uint d4) {
    float s = 0.0f;
    for (uint x = 0; x < d4; x++) {
        float4 k = float4(K[koff + 4 * x], K[koff + 4 * x + 1], K[koff + 4 * x + 2], K[koff + 4 * x + 3]);
        s += metal::dot(q[x], k);
    }
    return s;
}
"""

SOURCE = """
    uint r = thread_position_in_grid.x;              // query row: (batch * heads + head) * N + n
    uint n = uint(N[0]);
    uint D = uint(d[0]);
    uint d4 = D / 4;
    float scale = softmax_scale[0];
    uint base = (r / n) * n * D;                     // this (batch, head)'s K and V
    uint qoff = r * D;

    float4 q[32];
    for (uint x = 0; x < d4; x++) {
        q[x] = float4(Q[qoff + 4 * x], Q[qoff + 4 * x + 1], Q[qoff + 4 * x + 2], Q[qoff + 4 * x + 3]);
    }

    float mx = -INFINITY;
    for (uint j = 0; j < n; j++) mx = metal::max(mx, score(q, K, base + j * D, d4) * scale);

    float acc[128];
    for (uint x = 0; x < D; x++) acc[x] = 0.0f;
    float lsum = 0.0f;
    for (uint j0 = 0; j0 < n; j0 += 32) {
        float bacc[128];
        for (uint x = 0; x < D; x++) bacc[x] = 0.0f;
        float bl = 0.0f;
        uint j1 = metal::min(j0 + 32, n);
        for (uint j = j0; j < j1; j++) {
            float p = metal::precise::exp(score(q, K, base + j * D, d4) * scale - mx);
            bl += p;
            for (uint x = 0; x < D; x++) bacc[x] += p * V[base + j * D + x];
        }
        lsum += bl;
        for (uint x = 0; x < D; x++) acc[x] += bacc[x];
    }
    for (uint x = 0; x < D; x++) O_out[qoff + x] = acc[x] / lsum;
    l_out[r] = lsum;
    m_out[r] = mx;
"""
THREADGROUP = (64, 1, 1)


def grid(l):
    """One thread per query row; Metal dispatches exactly this many threads."""
    return (l.size, 1, 1)
