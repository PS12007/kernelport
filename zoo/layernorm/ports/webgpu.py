"""WebGPU (WGSL) port of layernorm: one workgroup per row, two-pass mean/variance in f32."""

WGSL = """
var<workgroup> red: array<f32, 256>;

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
    var s = 0.0;
    for (var c = tid; c < cols; c += 256u) { s += f32(x[base + c]); }
    let mean = block_sum(s, tid) / f32(cols);
    var q = 0.0;
    for (var c = tid; c < cols; c += 256u) {
        let d = f32(x[base + c]) - mean;
        q += d * d;
    }
    let rstd = 1.0 / sqrt(block_sum(q, tid) / f32(cols) + params.eps);
    for (var c = tid; c < cols; c += 256u) {
        y[base + c] = T((f32(x[base + c]) - mean) * rstd * f32(gamma[c]) + f32(beta[c]));
    }
}
"""


def grid(rows):
    return (rows,)
