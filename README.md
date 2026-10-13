# kernelport

Bring CUDA-only kernels to Mac, AMD, browsers, and CPU, with proof they are correct.

kernelport is an MCP server + agent skills + CLI. Your coding agent (Claude Code, Codex,
Cursor, Gemini CLI) writes the ports; kernelport runs them on real hardware and verifies
each one against a frozen, validated reference. A port is only ever reported as PASS if it
was verified on real hardware.

> Status: early development. See [PROGRESS.md](PROGRESS.md).
