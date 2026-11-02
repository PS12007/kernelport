"""docs/web (the in-browser WebGPU check) stays in step with the WebGPU ports it packages."""

import hashlib
import importlib.util
import json

import pytest

from conftest import ROOT
from kernelport.backends.webgpu.runner import check_browser_literals, load_port
from kernelport.harness.spec import LaunchSpec

WEB = ROOT / "docs" / "web"


def _build_script():
    spec = importlib.util.spec_from_file_location("build_web_demo", ROOT / "scripts" / "build_web_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (WEB / "data").is_dir(), reason="docs/web/data is not in the sdist")
def test_web_demo_matches_ports():
    """If this fails, run: python scripts/build_web_demo.py"""
    build = _build_script()
    manifest = json.loads((WEB / "cases.json").read_text(encoding="utf-8"))
    sources = build.sources()
    assert {s["group"] for s in sources} >= {"zoo", "llm.c", "flash-attention-minimal"}
    packaged = {(k["group"], k["name"]) for k in manifest["kernels"]}
    assert packaged == {(s["group"], s["name"]) for s in sources}, "docs/web is stale"
    for k in manifest["kernels"]:
        port = load_port(ROOT / k["port_file"])
        spec = LaunchSpec.load(ROOT / k["spec_file"])
        assert k["wgsl"] == build.wgsl_for(spec, port, k["dtype"]), f"{k['name']}: docs/web is stale"
        check_browser_literals(k["wgsl"])
        blob = (WEB / k["data"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == k["data_sha256"]
        for c in k["cases"]:
            for b in c["buffers"] + c["outputs"]:
                assert b["offset"] + b["size"] <= len(blob)
            assert all(1 <= g <= 65535 for g in c["grid"])


def test_web_page_is_self_contained():
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert "<script src" not in page and "@import" not in page  # no third-party code
    assert 'fetch("cases.json")' in page
