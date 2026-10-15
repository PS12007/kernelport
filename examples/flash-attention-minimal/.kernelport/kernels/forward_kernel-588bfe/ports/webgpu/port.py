"""WebGPU (WGSL) port of flash-attention-minimal's forward_kernel: one invocation per query row.

Like the CUDA kernel, each query row is one thread; there is no shared-memory tiling. Softmax in
two passes: the row max m first, then p = exp(s - m), the row sum l and O = (p V) / l. To keep
fp32 rounding small without fp64 (WGSL has none): scores are vec4 dot products (16 sequential adds
for d = 64 instead of 64), and sums are taken per block of 32 keys and then across blocks (not
Kahan summation, which fast-math compilers may remove). Supports head dims d <= 128 that are
multiples of 4; N need not be a multiple of 32. Writes the final O, l and m, which the original
leaves in its inout buffers.
"""

WGSL = """
fn score(q: ptr<function, array<vec4<f32>, 32>>, koff: u32, d4: u32) -> f32 {
    var s = 0.0;
    for (var x = 0u; x < d4; x++) {
        let k = vec4<f32>(K[koff + 4u * x], K[koff + 4u * x + 1u], K[koff + 4u * x + 2u], K[koff + 4u * x + 3u]);
        s += dot((*q)[x], k);
    }
    return s;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;                                  // query row: (batch * heads + head) * N + n
    if (r >= arrayLength(&l)) { return; }
    let N = u32(params.N);
    let D = u32(params.d);
    let d4 = D / 4u;
    let scale = params.softmax_scale;
    let base = (r / N) * N * D;                     // this (batch, head)'s K and V
    let qoff = r * D;

    var q: array<vec4<f32>, 32>;
    for (var x = 0u; x < d4; x++) {
        q[x] = vec4<f32>(Q[qoff + 4u * x], Q[qoff + 4u * x + 1u], Q[qoff + 4u * x + 2u], Q[qoff + 4u * x + 3u]);
    }

    var mx = -3.4028234e38;                         // rounds to -FLT_MAX
    for (var j = 0u; j < N; j++) { mx = max(mx, score(&q, base + j * D, d4) * scale); }

    var acc = array<f32, 128>();
    var lsum = 0.0;
    for (var j0 = 0u; j0 < N; j0 += 32u) {
        var bacc = array<f32, 128>();               // explicit zero: see check_loop_vars in the runner
        var bl = 0.0;
        for (var j = j0; j < min(j0 + 32u, N); j++) {
            let p = exp(score(&q, base + j * D, d4) * scale - mx);
            bl += p;
            for (var x = 0u; x < D; x++) { bacc[x] += p * V[base + j * D + x]; }
        }
        lsum += bl;
        for (var x = 0u; x < D; x++) { acc[x] += bacc[x]; }
    }
    for (var x = 0u; x < D; x++) { O[qoff + x] = acc[x] / lsum; }
    l[r] = lsum;
    m[r] = mx;
}
"""


def grid(l):
    """One invocation per query row (l has one element per row)."""
    return ((l.size + 63) // 64,)
