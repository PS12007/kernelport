"""WebGPU (WGSL) port of reduce_sum: out[0] += sum(x).

Each workgroup reduces a grid-stride slice in workgroup memory, then one
invocation adds the partial into out[0]. Core WGSL has no float atomics, so the
add is a compare-exchange loop on the float's bits (out is declared as
atomic<u32> through BINDINGS).
"""

BINDINGS = {"out": "array<atomic<u32>>"}

WGSL = """
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = u32(params.n);
    var acc = 0.0;
    for (var i = gid.x; i < n; i += nwg.x * 256u) { acc += f32(x[i]); }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let partial = red[0];
        var old = atomicLoad(&out[0]);
        loop {
            let r = atomicCompareExchangeWeak(&out[0], old, bitcast<u32>(bitcast<f32>(old) + partial));
            if (r.exchanged) { break; }
            old = r.old_value;
        }
    }
}
"""


def grid(n):
    return (min((n + 255) // 256, 1024),)
