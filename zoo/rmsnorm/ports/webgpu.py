"""WebGPU (WGSL) port of rmsnorm: one workgroup per row; y = x / sqrt(mean(x^2) + eps) * weight."""

WGSL = """
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let cols = u32(params.cols);
    let base = wid.x * cols;
    var sq = 0.0;
    for (var c = tid; c < cols; c += 256u) {
        let v = f32(x[base + c]);
        sq += v * v;
    }
    red[tid] = sq;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let inv_rms = 1.0 / sqrt(red[0] / f32(cols) + params.eps);
    for (var c = tid; c < cols; c += 256u) {
        y[base + c] = T(f32(x[base + c]) * inv_rms * f32(weight[c]));
    }
}
"""


def grid(rows):
    return (rows,)
