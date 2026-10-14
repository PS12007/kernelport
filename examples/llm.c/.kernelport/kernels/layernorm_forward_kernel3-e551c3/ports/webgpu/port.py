"""WebGPU (WGSL) port of llm.c layernorm_forward_kernel3: one workgroup per row, two passes in f32."""

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
    let cols = u32(params.C);
    let row = wid.x;
    let base = row * cols;
    var s = 0.0;
    for (var c = tid; c < cols; c += 256u) { s += inp[base + c]; }
    let m = block_sum(s, tid) / f32(cols);
    var q = 0.0;
    for (var c = tid; c < cols; c += 256u) {
        let d = inp[base + c] - m;
        q += d * d;
    }
    let r = 1.0 / sqrt(block_sum(q, tid) / f32(cols) + 1e-5);
    if (tid == 0u) {
        mean[row] = m;
        rstd[row] = r;
    }
    for (var c = tid; c < cols; c += 256u) {
        out[base + c] = r * (inp[base + c] - m) * weight[c] + bias[c];
    }
}
"""


def grid(N):
    return (N,)
