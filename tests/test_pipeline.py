"""Pipelines: kernels chained over shared buffers, each stage checked on its real input."""

import copy
import json
import shutil

import pytest

from conftest import BAD_PORTS, ROOT, needs_cuda, needs_torch, needs_triton

LLMC = ROOT / "examples" / "llm.c"
PIPE = LLMC / ".kernelport" / "pipelines" / "gpt2_loss_head.json"
CE = "crossentropy_forward_kernel1"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "llmc"
    shutil.copytree(LLMC, r, ignore=shutil.ignore_patterns("__pycache__", "results"))
    (r / ".git").mkdir()
    return r


def _pipe() -> dict:
    return json.loads(PIPE.read_text(encoding="utf-8"))


def _small(d: dict) -> dict:
    """The committed pipeline, cut to the small cases (fast on CPU)."""
    d = copy.deepcopy(d)
    d["cases"] = [c for c in d["cases"] if c["name"].startswith(("tiny", "odd_3", "confident"))]
    return d


# ----------------------------------------------------------------------------- spec checks
@needs_torch
def test_committed_pipeline_resolves(repo):
    from kernelport.harness.pipeline import PipelineSpec, resolve
    from kernelport.workspace import Workspace

    ws = Workspace(repo)
    stages = resolve(PipelineSpec.load(PIPE), ws.pipeline_kernel)
    assert [s.spec.name for s in stages] == ["softmax_forward_online_kernel8", CE,
                                            "crossentropy_softmax_backward_kernel1"]
    assert stages[0].bind == {"out": "probs", "inp": "logits"}
    assert stages[2].bind["dlogits"] == "dlogits" and stages[2].dtype == "float32"


@needs_torch
@pytest.mark.parametrize("edit, message", [
    (lambda d: d["stages"][0]["bind"].pop("inp"), "param inp is not bound"),
    (lambda d: d["stages"][0]["bind"].update(nope="logits"), "params the kernel does not have"),
    (lambda d: d["buffers"]["logits"].update(dtype="float16"), "param inp is float32 but buffer logits is float16"),
    (lambda d: d["buffers"]["logits"].pop("init"), "reads buffer logits before any stage writes it"),
    (lambda d: d["stages"].reverse(), "before any stage writes it"),
    (lambda d: d["stages"][0].update(dims={"N": "B", "C": "V"}), "has 3000 elements, buffer probs has 15000"),
    (lambda d: d["stages"][0].update(dims={}), "map the kernel's dims"),
    (lambda d: d["stages"][1].update(kernel="no_such_kernel"), "unknown kernel"),
    (lambda d: d["stages"][0]["bind"].update(out="nowhere"), "unknown buffer 'nowhere'"),
    (lambda d: d.update(name="../evil"), "pipeline name"),
    (lambda d: d.update(stages=[]), "no stages"),
    (lambda d: d.update(tolerance=1.0), "unknown pipeline keys"),
])
def test_write_pipeline_rejects_inconsistent_specs(repo, edit, message):
    from kernelport import tools

    d = _pipe()
    edit(d)
    out = tools.write_pipeline(d, root=str(repo))
    assert not out["ok"]
    assert message in out["summary"], out["summary"]


@needs_torch
def test_write_pipeline_saves_and_verify_refuses_missing_ports(repo):
    from kernelport import tools

    d = _pipe()
    d["name"] = "copy"
    out = tools.write_pipeline(d, root=str(repo))
    assert out["ok"], out
    assert (repo / ".kernelport" / "pipelines" / "copy.json").is_file()
    # crossentropy has no Metal port in the example
    res = tools.verify_pipeline("copy", "metal", root=str(repo))
    assert res["status"] == "REFUSED" and CE in res["summary"]
    assert tools.verify_pipeline("nope", "triton", root=str(repo))["status"] == "REFUSED"


@needs_torch
def test_cli_saves_a_json_file_and_reports(repo, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from kernelport.cli import app

    d = _pipe()
    d["name"] = "from_file"
    f = tmp_path / "p.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setenv("KERNELPORT_ROOT", str(repo))
    runner = CliRunner()
    r = runner.invoke(app, ["pipeline", str(f), "--backend", "metal"])  # no Metal port for crossentropy
    assert (repo / ".kernelport" / "pipelines" / "from_file.json").is_file()
    assert r.exit_code == 1 and "no metal port" in r.output, r.output
    bad = dict(d, stages=[])
    f.write_text(json.dumps(bad), encoding="utf-8")
    r = runner.invoke(app, ["pipeline", str(f)])
    assert r.exit_code == 1 and "no stages" in r.output, r.output


def test_no_pipeline_tool_accepts_tolerances():
    import inspect

    from kernelport import tools

    for fn in (tools.write_pipeline, tools.verify_pipeline):
        assert not any("tol" in p for p in inspect.signature(fn).parameters)


# ---------------------------------------------------------- running, without a GPU
def _run(repo, runner):
    from kernelport.harness.pipeline import PipelineSpec, resolve, run_pipeline
    from kernelport.workspace import Workspace

    ws = Workspace(repo)
    pipe = PipelineSpec.from_dict(_small(_pipe()))
    stages = resolve(pipe, ws.pipeline_kernel)
    return run_pipeline(pipe, stages, runner, backend="test", hardware="cpu", code_hashes={},
                        reference_hashes={})


def _reference_runner(rs, data):
    """A 'port' that is the reference itself, rounded to the kernel's dtypes."""
    from kernelport.harness.inputs import TORCH_DTYPES
    from kernelport.harness.reference import run_reference

    out = run_reference(rs.reference, rs.spec, data)
    return {p.name: out[p.name].to(TORCH_DTYPES[p.resolved_dtype(rs.dtype)])
            for p in rs.spec.params if p.is_output}


@needs_torch
def test_chain_of_correct_stages_passes_and_reports_drift(repo):
    res = _run(repo, _reference_runner)
    assert res.status == "PASS", res.failure_report()
    assert len(res.cases) == 3 and all(len(c.stages) == 3 for c in res.cases)
    lines = res.drift_lines()
    assert {line.split(":")[0] for line in lines} == {"probs", "losses", "dlogits"}
    # fp32 softmax underflows to 0 on confident logits; float64 does not, so the loss is inf in one chain
    assert any("inf/NaN" in line and "confident" in line for line in lines if line.startswith("losses"))


@needs_torch
def test_stage_bug_invisible_to_single_kernel_cases_fails_the_pipeline(repo):
    import torch

    def clamping(rs, data):
        out = _reference_runner(rs, data)
        if rs.spec.name == CE:  # the bug in tests/bad_ports/crossentropy_clamped.py
            p = torch.gather(data.tensors["probs"], 2, data.tensors["targets"].long().unsqueeze(-1)).squeeze(-1)
            out["losses"] = -torch.log(p.clamp_min(1e-3))
        return out

    res = _run(repo, clamping)
    assert res.status == "FAIL"
    report = res.failure_report()
    assert f"stage 2:{CE}: losses" in report
    assert "stage 3" not in report  # only the first failing stage of a case is reported


@needs_torch
def test_crashing_stage_is_reported_and_stops_the_case(repo):
    def crash(rs, data):
        if rs.index == 1:
            raise RuntimeError("boom")
        return _reference_runner(rs, data)

    res = _run(repo, crash)
    assert res.status == "FAIL"
    assert all(len(c.stages) == 2 and "boom" in c.stages[1].error for c in res.cases)
    assert all(not c.end_to_end for c in res.cases)


@needs_torch
def test_inputs_are_deterministic(repo):
    from kernelport.harness.pipeline import PipelineSpec, initial_buffers

    pipe = PipelineSpec.from_dict(_pipe())
    case = pipe.cases[1]
    a, ha = initial_buffers(pipe, case)
    b, hb = initial_buffers(pipe, case)
    assert ha == hb and all((a[k] == b[k]).all() or a[k].isnan().all() for k in a)
    assert int(a["targets"].max()) < case.dims["V"]
    assert a["probs"].isnan().all()  # written by a stage: starts as the NaN sentinel


@needs_torch
def test_result_round_trips(repo, tmp_path):
    from kernelport.harness.pipeline import PipelineResult

    res = _run(repo, _reference_runner)
    res.save(tmp_path / "r.json")
    back = PipelineResult.load(tmp_path / "r.json")
    assert back.status == "PASS" and back.drift_lines() == res.drift_lines()


# ---------------------------------------------------------------- on real hardware
@needs_triton
def test_triton_ports_pass_in_a_chain(repo):
    from kernelport import tools
    from kernelport.workspace import Workspace

    small = _small(_pipe())
    small["name"] = "small"
    assert tools.write_pipeline(small, root=str(repo))["ok"]
    res = tools.verify_pipeline("small", "triton", root=str(repo))
    assert res["status"] == "PASS", res
    assert len(Workspace(repo).pipeline_results("small")) == 1


@needs_triton
def test_clamped_port_passes_alone_but_fails_in_the_pipeline(repo):
    from kernelport import tools

    tools.write_port(CE, "triton", (BAD_PORTS / "crossentropy_clamped.py").read_text(), root=str(repo))
    alone = tools.verify_port(CE, "triton", root=str(repo))
    assert alone["status"] == "PASS", alone  # the kernel's own cases cannot see the clamp
    small = _small(_pipe())
    small["name"] = "small"
    tools.write_pipeline(small, root=str(repo))
    res = tools.verify_pipeline("small", "triton", root=str(repo))
    assert res["status"] == "FAIL"
    assert f"2:{CE}: losses" in res["report"]


@needs_cuda
def test_originals_agree_with_references_in_a_chain(repo):
    from kernelport import tools

    small = _small(_pipe())
    small["name"] = "small"
    tools.write_pipeline(small, root=str(repo))
    res = tools.verify_pipeline("small", "cuda", root=str(repo))
    assert res["status"] == "PASS", res
