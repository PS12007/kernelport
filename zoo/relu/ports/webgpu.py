"""WebGPU (WGSL) port of relu: y[i] = max(x[i], 0). Grid-stride loop (65535-workgroup cap)."""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    for (var i = gid.x; i < n; i += nwg.x * 256u) {
        let v = x[i];
        y[i] = select(T(0.0), v, v > T(0.0));   // CUDA: v > 0 ? v : 0
    }
}
"""


def grid(n):
    return (min((n + 255) // 256, 65535),)
