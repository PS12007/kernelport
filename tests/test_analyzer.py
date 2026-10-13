import json
from pathlib import Path

import pytest

from conftest import ZOO
from kernelport.analyzer.cuda_parser import scan, split_top_level, strip_comments_and_strings
from kernelport.analyzer.suggest import has_todo, safe_name, suggest_spec

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tricky():
    return {k.id: k for k in scan(FIX / "tricky.cu")}


def test_ignores_comments_strings_and_declarations(tricky):
    names = {k.name for k in tricky.values()}
    assert not {"fake_in_comment", "fake_in_block_comment", "fake_in_string", "declared_only"} & names
    assert "warp_reduce" not in names  # __device__, not __global__


def test_finds_all_definitions(tricky):
    assert sorted(k.name for k in tricky.values()) == [
        "multi_line_kernel", "overloaded", "overloaded", "passes_to_helper", "plain_c"]


def test_overloads_get_unique_ids(tricky):
    ids = [i for i in tricky if "overloaded" in i]
    assert len(ids) == len(set(ids)) == 2
    assert all("@" in i for i in ids)


def test_multi_line_template_kernel(tricky):
    k = tricky["tricky.cu::multi_line_kernel"]
    assert k.template_params == ["typename scalar_t", "int BLOCK = 256", "typename Acc = float"]
    assert k.launch_bounds == "BLOCK, 2"
    assert k.line == 18
    assert [p.name for p in k.params] == ["input", "output", "partial", "idx", "n", "scale"]
    kinds = {p.name: p.kind for p in k.params}
    assert kinds == {"input": "input", "output": "output", "partial": "inout",
                     "idx": "input", "n": "scalar", "scale": "scalar"}
    dtypes = {p.name: p.dtype for p in k.params}
    assert dtypes == {"input": "T", "output": "T", "partial": "float32", "idx": "int64",
                      "n": "int32", "scale": "float32"}
    for f in ("templates", "dynamic_shared_memory", "shared_memory", "syncthreads",
              "atomics", "warp_shuffle"):
        assert f in k.features, f
    assert k.source.lstrip().startswith("template")


def test_extern_c_and_vector_types(tricky):
    k = tricky["tricky.cu::plain_c"]
    assert k.extern_c
    assert {"vector_loads", "grid_stride_loop", "fp16"} <= set(k.features)
    data, h, _ = k.params
    assert data.kind == "inout" and data.base_type == "float4" and data.dtype is None
    assert h.kind == "input"


def test_deref_write_is_output(tricky):
    by_line = {k.line: k for k in tricky.values() if k.name == "overloaded"}
    assert by_line[46].params[0].kind == "output"  # a[0] = 1.0f
    assert by_line[47].params[0].kind == "output"  # *a = 2.0


def test_pointer_passed_to_function_is_inout(tricky):
    k = tricky["tricky.cu::passes_to_helper"]
    kinds = {p.name: p.kind for p in k.params}
    assert kinds == {"buf": "inout", "counter": "inout"}


def test_launch_sites(tricky):
    k = tricky["tricky.cu::multi_line_kernel"]
    assert len(k.launches) == 1
    l0 = k.launches[0]
    assert (l0.grid, l0.block, l0.shared_mem, l0.stream) == ("grid", "block", "256 * sizeof(float)", "s")
    assert l0.template_args == "float, 256"
    assert l0.args[-1] == "0.5f"
    pc = tricky["tricky.cu::plain_c"].launches[0]
    assert (pc.grid, pc.block) == ("(n + 127) / 128", "128")


def test_python_embedded_kernel():
    ks = scan(FIX / "embedded_kernel.py")
    assert [k.name for k in ks] == ["scale"]
    assert [p.kind for p in ks[0].params] == ["input", "output", "scalar", "scalar"]
    assert ks[0].line == 6  # line of __global__ in the .py file


def test_zoo_kinds_match_hand_written_specs():
    """The analyzer must agree with every hand-written zoo launch.json on param kinds and dtypes."""
    for k in scan(ZOO):
        spec = json.loads((ZOO / k.file.split("/")[0] / "launch.json").read_text())
        by_cuda = {p.get("cuda_name", p["name"]): p for p in spec["params"]}
        for p in k.params:
            sp = by_cuda.get(p.name) or by_cuda.get(safe_name(p.name))
            assert sp is not None, f"{k.id}: {p.name} not in spec"
            assert p.kind == sp["kind"], f"{k.id}: {p.name} kind {p.kind} != {sp['kind']}"
            assert p.dtype == sp["dtype"], f"{k.id}: {p.name} dtype {p.dtype} != {sp['dtype']}"


def test_zoo_scan_finds_all_twelve():
    assert len(scan(ZOO)) == 12


def test_strip_preserves_offsets():
    src = 'a // x\n"str" /* c\n d */ b'
    out = strip_comments_and_strings(src)
    assert len(out) == len(src) and out.count("\n") == src.count("\n")
    assert out.split("\n")[0].strip() == "a"
    assert out.endswith(" b")


def test_split_top_level():
    assert split_top_level("const Foo<int, 3>* a, int b[2], f(x, y)") == ["const Foo<int, 3>* a", "int b[2]", "f(x, y)"]
    split_top_level("a < b, c > d")  # ambiguous in C++; must not crash


def test_suggest_spec_1d_kernel_is_complete():
    k = next(k for k in scan(ZOO / "vector_add"))
    spec = suggest_spec(k)
    assert spec["entry"] == "vector_add<{T}>"
    assert spec["launch"]["grid"] == ["TODO"]  # no launch site in the zoo file
    spec["launch"] = {"grid": ["cdiv(n, 256)"], "block": [256]}
    assert not has_todo(spec), spec
    from kernelport.harness.spec import LaunchSpec

    LaunchSpec.from_dict(spec)  # everything else is already a loadable spec


def test_suggest_spec_marks_unknowns():
    k = next(k for k in scan(ZOO / "transpose"))
    spec = suggest_spec(k)
    assert has_todo(spec)
    names = [p["name"] for p in spec["params"]]
    assert "in_" in names  # python keyword renamed
    assert spec["params"][0]["cuda_name"] == "in"


def test_suggest_spec_template_defaults():
    k = scan(FIX / "tricky.cu")[0]
    assert suggest_spec(k)["entry"] == "multi_line_kernel<{T}, 256, float>"


def test_safe_name():
    assert safe_name("in") == "in_"
    assert safe_name("x") == "x"
    assert safe_name("lambda") == "lambda_"
