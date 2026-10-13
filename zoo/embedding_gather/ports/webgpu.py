"""WebGPU (WGSL) port of embedding_gather: out[t, :] = table[ids[t], :]; one workgroup per token.

ids is int64 -> vec2<u32>; the row index is the low word .x.
"""

WGSL = """
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let t = wid.x;
    let dim = u32(params.dim);
    let src = ids[t].x * dim;
    let dst = t * dim;
    for (var d = tid; d < dim; d += 128u) { out[dst + d] = table[src + d]; }
}
"""


def grid(tokens):
    return (tokens,)
