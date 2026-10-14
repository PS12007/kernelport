"""Talk to the real MCP server over stdio, as an agent would."""

import asyncio
import json
import os
import sys

from conftest import ROOT

EXPECTED_TOOLS = {"kernelport_doctor", "scan_kernels", "read_kernel", "read_skill", "write_reference",
                  "validate_reference", "write_port", "verify_port", "bench_port", "tune_port", "list_status", "make_report"}


async def _session(repo, fn):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {**os.environ, "KERNELPORT_ROOT": str(repo), "PYTHONPATH": str(ROOT)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "kernelport.mcp_server"], env=env, cwd=str(repo))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def _payload(result):
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


def test_lists_all_tools_without_root_param(tmp_path):
    async def go(s):
        return (await s.list_tools()).tools

    listed = asyncio.run(_session(tmp_path, go))
    assert {t.name for t in listed} == EXPECTED_TOOLS
    for t in listed:
        assert "root" not in t.input_schema.get("properties", {}), t.name
        assert t.description


def test_scan_read_status_over_protocol(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "k.cu").write_text((ROOT / "zoo" / "relu" / "kernel.cu").read_text())

    async def go(s):
        scan = _payload(await s.call_tool("scan_kernels", {"path": "."}))
        read = _payload(await s.call_tool("read_kernel", {"kernel_id": "relu"}))
        skill = _payload(await s.call_tool("read_skill", {"name": "workflow"}))
        status = _payload(await s.call_tool("list_status", {}))
        refused = _payload(await s.call_tool("verify_port", {"kernel_id": "relu", "backend": "triton"}))
        return scan, read, skill, status, refused

    scan, read, skill, status, refused = asyncio.run(_session(tmp_path, go))
    assert scan["ok"] and scan["kernels"][0]["kernel_id"] == "k.cu::relu"
    assert "__global__" in read["source"]
    assert skill["ok"]
    assert status["rows"][0]["status"] == "SKIPPED"
    assert refused["ok"] is False and refused["status"] == "REFUSED"
    assert (tmp_path / ".kernelport" / "index.json").is_file()
