"""WebGPU (WGSL) port of saxpy, in place: y = a * x + y, computed in f32."""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    for (var i = gid.x; i < n; i += nwg.x * 256u) {
        y[i] = T(params.a * f32(x[i]) + f32(y[i]));   // y is inout (read_write)
    }
}
"""


def grid(n):
    return (min((n + 255) // 256, 65535),)
