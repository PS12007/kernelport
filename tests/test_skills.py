"""Skills must stay consistent with the tools and the spec format."""

import re

from kernelport import tools
from kernelport.harness.spec import PARAM_KINDS


def _skill(name):
    res = tools.read_skill(name)
    assert res["ok"], res
    return res["content"]


def test_skill_frontmatter():
    text = _skill("workflow")
    m = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert m and m.group(1) == "kernelport" and len(m.group(2)) > 40


def test_skill_mentions_every_tool():
    text = _skill("workflow")
    for fn in tools.TOOLS:
        assert f"{fn.__name__}(" in text, f"SKILL.md does not explain {fn.__name__}"


def test_skill_documents_every_param_kind():
    text = _skill("workflow")
    for kind in PARAM_KINDS:
        assert f"`{kind}`" in text


def test_skill_states_the_rules():
    text = _skill("workflow").lower()
    for phrase in ("never edit", "fix the reference", "6 failed", "pass"):
        assert phrase in text


def test_triton_guide_and_common_bugs_exist():
    tri = _skill("triton")
    for concept in ("program_id", "__syncthreads", "shuffle", "atomic", "cumsum", "fp16", "input_precision", "mask"):
        assert concept in tri, concept
    assert "unexpected NaN" in _skill("common_bugs")


def test_unwritten_backend_guides_report_cleanly():
    res = tools.read_skill("hip")
    assert res["ok"] or "not written yet" in res["error"]


def test_cpu_guide_covers_the_traps():
    cpu = _skill("cpu")
    for concept in ("prange", "in place", "inout", "atomic", "fastmath", "float16", "data race", "run("):
        assert concept in cpu, concept


def test_webgpu_guide_covers_the_traps():
    w = _skill("webgpu")
    for concept in ("workgroupBarrier", "65535", "vec2<u32>", "BINDINGS", "atomicCompareExchangeWeak",
                    "alias T", "params.", "fn main", "grid("):
        assert concept in w, concept


def test_metal_and_hip_guides_exist_and_state_their_status():
    m, h = _skill("metal"), _skill("hip")
    for concept in ("thread_position_in_grid", "threadgroup_barrier", "_out", "n[0]", "precise", "TOTAL threads"):
        assert concept in m, concept
    for concept in ("__hip_bfloat16", "warpSize", "64", "hipRTC"):
        assert concept in h, concept
    assert "not yet" in m and "not yet" in h  # honest about unverified status
