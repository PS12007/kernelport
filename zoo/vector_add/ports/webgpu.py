"""WebGPU (WGSL) port of vector_add: out[i] = a[i] + b[i].

Grid-stride loop: WebGPU caps a dispatch at 65535 workgroups per dimension,
so 16M elements cannot get one invocation each from a 1-D grid.
"""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    let stride = nwg.x * 256u;
    for (var i = gid.x; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
"""


def grid(n):
    return (min((n + 255) // 256, 65535),)
