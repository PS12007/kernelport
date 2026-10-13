"""WebGPU (WGSL) port of cumsum: inclusive prefix sum per row, one workgroup per row.

Same structure as the CUDA kernel: BLOCK-sized chunks scanned in workgroup
memory (Hillis-Steele), offset by a running carry.
"""

WGSL = """
var<workgroup> buf: array<f32, 256>;
var<workgroup> carry: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let cols = u32(params.cols);
    let base = wid.x * cols;
    if (tid == 0u) { carry = 0.0; }
    workgroupBarrier();
    for (var start = 0u; start < cols; start += 256u) {
        let c = start + tid;
        buf[tid] = select(0.0, f32(x[base + min(c, cols - 1u)]), c < cols);
        workgroupBarrier();
        for (var off = 1u; off < 256u; off <<= 1u) {
            var add = 0.0;
            if (tid >= off) { add = buf[tid - off]; }
            workgroupBarrier();
            buf[tid] += add;
            workgroupBarrier();
        }
        if (c < cols) { y[base + c] = T(carry + buf[tid]); }
        workgroupBarrier();
        if (tid == 0u) { carry += buf[255]; }
        workgroupBarrier();
    }
}
"""


def grid(rows):
    return (rows,)
