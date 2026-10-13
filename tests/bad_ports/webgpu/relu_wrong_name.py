"""BAD webgpu port: uses a buffer name that is not a launch.json param (shader compile error)."""

WGSL = """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < u32(params.n)) { output[i] = max(x[i], T(0.0)); }
}
"""


def grid(n):
    return (min((n + 255) // 256, 65535),)
