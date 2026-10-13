"""BAD webgpu port: asks for more than 65535 workgroups on the large case."""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < u32(params.n)) { out[i] = a[i] + b[i]; }
}
"""


def grid(n):
    return ((n + 255) // 256,)
