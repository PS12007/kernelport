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
    k7, = got["softmax_forward_kernel7"]
    assert k7.startswith("line 413: `if (tid >= C)`") and "`warpReduceMax` at line 432" in k7
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
    assert "softmax_forward_kernel7" not in warned(fixed)
    assert "softmax_forward_kernel7" in warned(src)


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
