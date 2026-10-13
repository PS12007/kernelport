"""WebGPU (WGSL) port of transpose: out[c, r] = inp[r, c].

32x8 workgroups each move a 32x32 tile through workgroup memory (+1 padding),
exactly like the CUDA kernel's shared-memory tile.
"""

WGSL = """
var<workgroup> tile: array<array<T, 33>, 32>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = u32(params.rows);
    let cols = u32(params.cols);
    var x = wid.x * 32u + lid.x;            // input column
    var y = wid.y * 32u + lid.y;            // input row
    for (var j = 0u; j < 32u; j += 8u) {
        if (x < cols && y + j < rows) { tile[lid.y + j][lid.x] = inp[(y + j) * cols + x]; }
    }
    workgroupBarrier();
    x = wid.y * 32u + lid.x;                // output column (= input row)
    y = wid.x * 32u + lid.y;                // output row (= input column)
    for (var j = 0u; j < 32u; j += 8u) {
        if (x < rows && y + j < cols) { out[(y + j) * rows + x] = tile[lid.x][lid.y + j]; }
    }
}
"""


def grid(rows, cols):
    return ((cols + 31) // 32, (rows + 31) // 32)
