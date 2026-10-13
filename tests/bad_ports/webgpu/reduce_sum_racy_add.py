"""BAD webgpu port: every workgroup does a plain read-modify-write of out[0] (no atomics).

Partials from concurrent workgroups overwrite each other on large inputs.
"""

WGSL = """
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    var acc = 0.0;
    for (var i = gid.x; i < n; i += nwg.x * 256u) { acc += f32(x[i]); }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { out[0] = out[0] + red[0]; }
}
"""


def grid(n):
    return (min((n + 255) // 256, 1024),)
