"""docs/web (the in-browser WebGPU check) stays in step with the zoo's WebGPU ports."""

import hashlib
import json

import pytest

from conftest import ROOT, ZOO
from kernelport.backends.webgpu.runner import bindings_source, check_browser_literals, load_port
from kernelport.harness.spec import LaunchSpec

WEB = ROOT / "docs" / "web"


@pytest.mark.skipif(not (WEB / "data").is_dir(), reason="docs/web/data is not in the sdist")
def test_web_demo_matches_zoo_ports():
    """If this fails, run: python scripts/build_web_demo.py"""
    manifest = json.loads((WEB / "cases.json").read_text(encoding="utf-8"))
    zoo = sorted(p.parent.parent.name for p in ZOO.glob("*/ports/webgpu.py"))
    for dtype in ("float32", "float16"):
        assert [k["name"] for k in manifest["kernels"] if k["dtype"] == dtype] == zoo
    for k in manifest["kernels"]:
        port = load_port(ZOO / k["name"] / "ports" / "webgpu.py")
        spec = LaunchSpec.load(ZOO / k["name"] / "launch.json")
        bindings = bindings_source(spec, k["dtype"], dict(getattr(port, "BINDINGS", {}) or {}))
        expected = ("enable f16; " if k["dtype"] == "float16" else "") + port.WGSL + "\n" + bindings
        assert k["wgsl"] == expected, f"{k['name']}: docs/web is stale"
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
