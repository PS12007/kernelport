"""Static warnings (analyzer/hazards.py) and brace matching across #if branches."""

from pathlib import Path

from kernelport import tools
from kernelport.analyzer.cuda_parser import parse_source, scan

ROOT = Path(__file__).resolve().parents[1]
LLMC = ROOT / "examples" / "llm.c" / "dev" / "cuda"


def warned(src: str) -> dict[str, list[str]]:
    return {k.name: k.warnings for k in parse_source(src, "k.cu") if k.warnings}


def test_llmc_softmax_kernels_flagged():
    # llm.c's kernel7 and online_kernel8 return early on tid >= C before full-mask warp
    # reductions; nothing else in llm.c's dev/cuda does.
    kernels = scan(LLMC)
    got = {k.name: k.warnings for k in kernels if k.warnings}
    assert sorted(got) == ["softmax_forward_kernel7", "softmax_forward_online_kernel8"]
    k7, k7_block = got["softmax_forward_kernel7"]
    assert k7.startswith("line 413: `if (tid >= C)`") and "`warpReduceMax` at line 432" in k7
    # its early return also skips a __syncthreads: fine on NVIDIA, a porting note
    assert "`__syncthreads` at line 435" in k7_block and "WGSL rejects" in k7_block
    k8, = got["softmax_forward_online_kernel8"]
    assert k8.startswith("line 513: `if (tid >= C)`") and "`__syncwarp` at line 544" in k8


def test_fixed_kernel7_not_flagged():
    src = (LLMC / "softmax_forward.cu").read_text(encoding="utf-8")
    old = """    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }"""
    assert old in src
    fixed = src.replace(old, """    if (warpId * 32 >= C) {
        if (laneId == 0) {
            maxvals[warpId] = -INFINITY;
            sumvals[warpId] = 0.0f;
        }
        return;
    }""")
    def warp_warnings(text):
        return [w for w in warned(text).get("softmax_forward_kernel7", []) if "warp-wide" in w]
    assert warp_warnings(fixed) == []
    assert len(warp_warnings(src)) == 1


GLOBAL_IDX = """
__global__ void k(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    v += __shfl_down_sync(0xffffffff, v, 16);
    out[i] = v;
}
"""


def test_global_index_exit_before_shuffle():
    w, = warned(GLOBAL_IDX)["k"]
    assert "`if (i >= n)`" in w and "`__shfl_down_sync` at line 6" in w


def test_partial_mask_not_flagged():
    assert warned(GLOBAL_IDX.replace("0xffffffff", "__activemask()")) == {}


def test_return_after_shuffle_not_flagged():
    src = """
__global__ void k(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? x[i] : 0.0f;
    v += __shfl_down_sync(0xffffffff, v, 16);
    if (i >= n) return;
    out[i] = v;
}
"""
    assert warned(src) == {}


def test_warp_uniform_exit_not_flagged():
    src = """
__global__ void k(float* out, const float* x, int rows, int C) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp >= rows) return;
    float v = 0.0f;
    for (int i = lane; i < C; i += 32) v += x[warp * C + i];
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    if (lane == 0) out[warp] = v;
}
"""
    assert warned(src) == {}


def test_helper_and_cg_reduce():
    helper = """
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}
__device__ float row_sum(float v) { return warp_sum(v); }
__global__ void a(float* out, int n) {
    int tid = threadIdx.x;
    if (tid >= n) { return; }
    out[0] = row_sum(1.0f);
}
__global__ void b(float* out, int n) {
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane = warp.thread_rank();
    if (lane >= n) return;
    out[0] = cg::reduce(warp, 1.0f, cg::plus<float>{});
}
"""
    got = warned(helper)
    assert sorted(got) == ["a", "b"]
    assert "`row_sum`" in got["a"][0] and "`cg::reduce`" in got["b"][0]


def test_if_branches_with_own_braces():
    # one brace opened per #if branch (llama.cpp's mmf.cuh does this); scan used to crash
    src = """
__global__ void k(float* y, int n) {
#if defined(A)
    if constexpr (n_is_small) {NO_DEVICE_CODE;} else {
    y[0] = 1.0f;
#elif defined(B)
    if constexpr (other) {NO_DEVICE_CODE;} else {
    y[0] = 2.0f;
#else
    {
    y[0] = 3.0f;
#endif
    }
}

__global__ void after(float* y) { y[0] = 4.0f; }
"""
    ks = parse_source(src, "k.cu")
    assert [(k.name, k.line, k.end_line) for k in ks] == [("k", 2, 14), ("after", 16, 16)]


def test_unbalanced_kernel_skipped_not_fatal():
    src = "__global__ void broken(float* y) { y[0] = 1.0f;\n__global__ void ok(float* y) { y[0] = 2.0f; }\n"
    assert [k.name for k in parse_source(src, "k.cu")] == ["ok"]


def test_tools_report_warnings(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "k.cu").write_text(GLOBAL_IDX, encoding="utf-8")
    res = tools.scan_kernels(".", root=str(repo))
    row, = res["kernels"]
    assert len(row["warnings"]) == 1 and "warnings for k" in res["summary"]
    read = tools.read_kernel(row["kernel_id"], root=str(repo))
    assert read["warnings"] == row["warnings"] and "1 warning(s)" in read["summary"]


# ---- block barriers ----------------------------------------------------------------------

def test_exit_before_syncthreads():
    src = """
__global__ void k(float* out, const float* x, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    s[threadIdx.x] = x[i];
    __syncthreads();
    out[i] = s[255 - threadIdx.x];
}
"""
    w, = warned(src)["k"]
    assert w.startswith("line 5: `if (i >= n)` lets some threads of the block return")
    assert "`__syncthreads` at line 7" in w and "WGSL rejects" in w


def test_guarded_work_not_flagged():
    src = """
__global__ void k(float* out, const float* x, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = i < n ? x[i] : 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int j = 0; j < 256; j++) t += s[j]; out[blockIdx.x] = t; }
    if (i >= n) return;
}
"""
    assert warned(src) == {}


def test_block_uniform_exit_and_loop_not_flagged():
    src = """
__global__ void k(float* out, const float* x, int rows, int n) {
    __shared__ float tile[32][32];
    if (blockIdx.x >= rows) return;
    for (int t = 0; t < (n + 31) / 32; ++t) {
        tile[threadIdx.y][threadIdx.x] = x[t * 32 + threadIdx.x];
        __syncthreads();
        out[t] += tile[threadIdx.x][threadIdx.y];
        __syncthreads();
    }
}
"""
    assert warned(src) == {}


def test_barrier_in_divergent_branch_and_loop():
    src = """
__global__ void branch(float* out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = 1.0f;
        __syncthreads();
    }
}
__global__ void loop(float* out, int N) {
    int warp = threadIdx.x / 32;  // the same for a warp, not for the block
    for (int row = warp; row < N; row += 8) {
        out[row] = 0.0f;
        __syncthreads();
    }
}
"""
    got = warned(src)
    assert "sits inside `if (tid < n)`" in got["branch"][0]
    assert "inside a loop whose condition `row < N`" in got["loop"][0]


def test_loop_shifted_by_thread_has_same_trip_count():
    # llama.cpp wkv.cu: start and bound both add tid, so every thread loops equally often
    src = """
__global__ void k(float* out, int T, int C, int base) {
    const int tid = threadIdx.x;
    for (int t = base + tid; t < base + T * C + tid; t += C) {
        out[t] = 0.0f;
        __syncthreads();
    }
}
"""
    assert warned(src) == {}


def test_exit_inside_block_that_always_returns():
    # llama.cpp mmq.cuh fixup: the early return is inside a branch that returns anyway
    src = """
__global__ void k(float* out, const int* ids, int n) {
    __shared__ int s[32];
    int i = threadIdx.x;
    if (!ids) {
        if (i > n) {
            return;
        }
        out[i] += 1.0f;
        return;
    }
    s[i] = ids[i];
    __syncthreads();
    out[i] = s[31 - i];
}
"""
    assert warned(src) == {}


def test_barrier_helpers_and_block_sync():
    src = """
template <float (*warp_red)(float)>
__device__ float blockReduce(float v) { __shared__ float s[32]; __syncthreads(); return v; }
__device__ float row_total(float v) { return blockReduce<warp_sum>(v); }
__global__ void a(float* out, int C) {
    int c = threadIdx.x;
    if (c >= C) return;
    out[0] = row_total(1.0f);
}
__global__ void b(float* out, int n) {
    cg::thread_block block = cg::this_thread_block();
    if (block.thread_rank() >= n) return;
    block.sync();
}
"""
    got = warned(src)
    assert sorted(got) == ["a", "b"]
    assert "`row_total`" in got["a"][0] and "`block.sync`" in got["b"][0]
