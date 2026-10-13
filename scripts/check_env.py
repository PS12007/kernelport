"""Print the environment report (same as `kernelport doctor`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernelport.doctor import print_report, run_doctor  # noqa: E402

if __name__ == "__main__":
    report = run_doctor()
    print_report(report)
    sys.exit(0 if report.ok else 1)
