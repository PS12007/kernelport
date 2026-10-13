import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ZOO = ROOT / "zoo"
sys.path.insert(0, str(ROOT))


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and not getattr(torch.version, "hip", None)
    except ImportError:
        return False


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


needs_cuda = pytest.mark.skipif(not (_cuda_ok() and _has("cupy")), reason="needs NVIDIA GPU + cupy")
needs_triton = pytest.mark.skipif(not (_cuda_ok() and _has("triton")), reason="needs GPU + triton")
needs_torch = pytest.mark.skipif(not _has("torch"), reason="needs torch (kernelport[ref])")
