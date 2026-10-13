"""BAD webgpu port: one element per invocation with the grid capped at 65535 workgroups.

Correct up to 65535*256 elements; the 16M case leaves the tail unwritten.
"""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < u32(params.n)) { out[i] = a[i] + b[i]; }
}
"""


def grid(n):
    return (min((n + 255) // 256, 65535),)
