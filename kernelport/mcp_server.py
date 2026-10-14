"""kernelport MCP server (stdio). Started by agents via ``kernelport mcp``.

Each MCP tool is a thin wrapper over kernelport.tools, without the ``root``
argument: the workspace is always the repo the server was started in
($KERNELPORT_ROOT, or the nearest ancestor of the cwd with .kernelport/ or .git).
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from kernelport import __version__, tools

INSTRUCTIONS = """kernelport ports CUDA kernels to other backends and PROVES each port correct.
You write references and ports; kernelport runs them on real hardware.
Start with read_skill("workflow") and follow it. Never claim a port works unless
verify_port returned PASS."""


def build_server() -> MCPServer:
    """Create the server with every kernelport tool registered."""
    server = MCPServer(name="kernelport", instructions=INSTRUCTIONS, version=__version__)

    @server.tool()
    def kernelport_doctor() -> dict[str, Any]:
        """Environment and available backends, each with a reason if unavailable."""
        return tools.kernelport_doctor()

    @server.tool()
    def scan_kernels(path: str = ".") -> dict[str, Any]:
        """Find CUDA kernels under a path in this repo; returns kernel ids with file and line."""
        return tools.scan_kernels(path)

    @server.tool()
    def read_kernel(kernel_id: str) -> dict[str, Any]:
        """Kernel source, parsed signature, launch sites, and the input spec (saved, or a suggestion to fill in)."""
        return tools.read_kernel(kernel_id)

    @server.tool()
    def read_skill(name: str = "workflow") -> dict[str, Any]:
        """Read a skill: 'workflow' (start here), 'common_bugs', or a backend guide: triton, cpu, metal, webgpu, hip."""
        return tools.read_skill(name)

    @server.tool()
    def write_reference(kernel_id: str, code: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save a candidate plain-PyTorch reference (must define `def reference(...)`), and optionally the input spec (launch.json dict)."""
        return tools.write_reference(kernel_id, code, spec)

    @server.tool()
    def validate_reference(kernel_id: str) -> dict[str, Any]:
        """Run the ORIGINAL CUDA kernel against the candidate reference. PASS freezes it; FAIL means fix the reference."""
        return tools.validate_reference(kernel_id)

    @server.tool()
    def write_port(kernel_id: str, backend: str, code: str) -> dict[str, Any]:
        """Save a port for a backend (triton, cpu, metal, webgpu, hip)."""
        return tools.write_port(kernel_id, backend, code)

    @server.tool()
    def verify_port(kernel_id: str, backend: str) -> dict[str, Any]:
        """Verify the saved port against the frozen reference on every input case: PASS, FAIL with report, or UNVERIFIED."""
        return tools.verify_port(kernel_id, backend)

    @server.tool()
    def bench_port(kernel_id: str, backend: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Time original vs port on this machine (port must have PASSED here). config: {"dtype", "case"}."""
        return tools.bench_port(kernel_id, backend, config)

    @server.tool()
    def tune_port(kernel_id: str, backend: str, dtype: str = "float32") -> dict[str, Any]:
        """After PASS: try every combination in the port's TUNE dict, verify each, keep the fastest passing one."""
        return tools.tune_port(kernel_id, backend, dtype)

    @server.tool()
    def list_status() -> dict[str, Any]:
        """Table of kernel x backend x hardware statuses."""
        return tools.list_status()

    @server.tool()
    def make_report(format: str = "html") -> dict[str, Any]:
        """Write the report: format "html" (returns its path) or "markdown" (also returns the text as content)."""
        return tools.make_report(format)

    return server


def main() -> None:
    """Run over stdio."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
