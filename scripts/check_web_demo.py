"""Run docs/web/index.html in a headless Chromium browser (Chrome or Edge) and print its result.

    python scripts/check_web_demo.py [path-to-chrome-or-msedge] [--timeout 180]

Serves docs/web on 127.0.0.1, opens the page with ?post=1 so it POSTs its result back,
and exits non-zero unless every port passes. The browser picks its own WebGPU adapter.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "docs" / "web"
CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]


def _browser(given: str | None) -> str:
    for c in ([given] if given else CANDIDATES):
        found = shutil.which(c) or (c if Path(c).is_file() else None)
        if found:
            return found
    sys.exit("no Chrome/Edge found; pass its path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("browser", nargs="?")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--dir", type=Path, default=WEB, help="page directory to serve (default docs/web)")
    ap.add_argument("--discrete", action="store_true", help="ask Chromium for the high-performance GPU")
    args = ap.parse_args()

    got: dict = {}
    done = threading.Event()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            got.update(json.loads(body or b"{}"))
            self.send_response(204)
            self.end_headers()
            done.set()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(args.dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/?post=1"
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen([_browser(args.browser), "--headless=new", "--enable-unsafe-webgpu",
                                 f"--user-data-dir={profile}", "--no-first-run",
                                 *(["--force_high_performance_gpu"] if args.discrete else []), url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            finished = done.wait(args.timeout)
        finally:
            proc.kill()
            proc.wait()
            server.shutdown()
    if not finished:
        sys.exit(f"no result within {args.timeout:.0f} s")
    if "error" in got:
        sys.exit(f"page error: {got['error']}")
    for line in got["kernels"]:
        print("  " + line.replace(" ▶ WGSL port", "").replace(" WGSL port", ""))
    print(f"{got['passed']}/{got['total']} ports PASS ({got['cases']} cases) on {got['adapter']}")
    sys.exit(0 if got["passed"] == got["total"] else 1)


if __name__ == "__main__":
    main()
