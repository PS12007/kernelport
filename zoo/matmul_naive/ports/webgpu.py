"""WebGPU (WGSL) port of matmul: C[M, N] = A[M, K] @ B[K, N]; one invocation per output, f32 sum."""

WGSL = """
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = u32(params.M);
    let N = u32(params.N);
    let K = u32(params.K);
    let col = gid.x;
    let row = gid.y;
    if (row < M && col < N) {
        var acc = 0.0;
        for (var k = 0u; k < K; k++) { acc += f32(A[row * K + k]) * f32(B[k * N + col]); }
        C[row * N + col] = T(acc);
    }
}
"""


def grid(M, N):
    return ((N + 15) // 16, (M + 15) // 16)
