"""Metal (MLX) port of transpose: 32x32 tile in threadgroup memory, 32x8 threadgroups (as in CUDA)."""

SOURCE = """
    threadgroup T tile[32][33];
    uint R = uint(rows[0]);
    uint C = uint(cols[0]);
    uint2 tg = threadgroup_position_in_grid.xy;
    uint2 lid = thread_position_in_threadgroup.xy;
    uint x = tg.x * 32 + lid.x;              // input column
    uint y = tg.y * 32 + lid.y;              // input row
    for (uint j = 0; j < 32; j += 8) {
        if (x < C && y + j < R) tile[lid.y + j][lid.x] = inp[(y + j) * C + x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x = tg.y * 32 + lid.x;                   // output column (= input row)
    y = tg.x * 32 + lid.y;                   // output row (= input column)
    for (uint j = 0; j < 32; j += 8) {
        if (x < R && y + j < C) out[(y + j) * R + x] = tile[lid.x][lid.y + j];
    }
"""
THREADGROUP = (32, 8, 1)


def grid(rows, cols):
    # total threads: one 32x8 threadgroup per 32x32 tile
    return (((cols + 31) // 32) * 32, ((rows + 31) // 32) * 8, 1)
