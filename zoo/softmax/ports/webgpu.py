"""WebGPU (WGSL) port of row-wise softmax: one workgroup per row, like the CUDA kernel.

Max, sum of exp(x - max), normalise; block reductions in workgroup memory.
"""

WGSL = """
var<workgroup> red: array<f32, 256>;

fn block_max(v: f32, tid: u32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn block_sum(v: f32, tid: u32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let cols = u32(params.cols);
    let base = wid.x * cols;
    var m = -3.4028235e38;                  // WGSL has no inf literal; -FLT_MAX is the identity here
    for (var c = tid; c < cols; c += 256u) { m = max(m, f32(x[base + c])); }
    m = block_max(m, tid);
    var s = 0.0;
    for (var c = tid; c < cols; c += 256u) { s += exp(f32(x[base + c]) - m); }
    let inv = 1.0 / block_sum(s, tid);
    for (var c = tid; c < cols; c += 256u) { y[base + c] = T(exp(f32(x[base + c]) - m) * inv); }
}
"""


def grid(rows):
    return (rows,)
