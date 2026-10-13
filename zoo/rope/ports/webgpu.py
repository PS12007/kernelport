"""WebGPU (WGSL) port of rope (interleaved pairs): one workgroup per token.

positions is int64 in launch.json; WGSL has no 64-bit ints, so it arrives as
vec2<u32> and the (non-negative, small) position is the low word .x.
"""

WGSL = """
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let t = wid.x;
    let heads = u32(params.heads);
    let head_dim = u32(params.head_dim);
    let half = head_dim / 2u;
    let p = positions[t].x;
    for (var j = tid; j < heads * half; j += 128u) {
        let h = j / half;
        let i = j % half;
        let base = (t * heads + h) * head_dim + 2u * i;
        let x0 = f32(x[base]);
        let x1 = f32(x[base + 1u]);
        let c = cos_cache[p * half + i];
        let s = sin_cache[p * half + i];
        y[base] = T(x0 * c - x1 * s);
        y[base + 1u] = T(x0 * s + x1 * c);
    }
}
"""


def grid(tokens):
    return (tokens,)
