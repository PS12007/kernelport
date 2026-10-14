"""WebGPU (WGSL) port of llm.c softmax_forward_online_kernel8: one workgroup per row.

Three passes in f32 (max, sum of exp(x - max), normalise). WGSL has no infinity literal and
exp(-inf) is implementation-defined, so the running max starts at the lowest finite f32.
Any C >= 1 works (the original needs C >= blockDim.x).
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
    let cols = u32(params.C);
    let base = wid.x * cols;
    var m = -3.4028234e38;
    for (var c = tid; c < cols; c += 256u) { m = max(m, inp[base + c]); }
    let row_max = block_max(m, tid);
    var s = 0.0;
    for (var c = tid; c < cols; c += 256u) { s += exp(inp[base + c] - row_max); }
    let row_sum = block_sum(s, tid);
    for (var c = tid; c < cols; c += 256u) { out[base + c] = exp(inp[base + c] - row_max) / row_sum; }
}
"""


def grid(N):
    return (N,)
