#!/usr/bin/env bash
# Verify kernelport's AMD paths on a ROCm machine (e.g. an AMD Developer Cloud MI300X droplet
# with the ROCm image). Checks two things nobody has run on AMD yet:
#   1. the hip backend: 12 zoo HIP ports compiled with hipRTC (via ROCm CuPy)
#   2. the triton backend on ROCm: 12 zoo Triton ports + 9 llm.c Triton ports
# Originals cannot run here (no NVIDIA); ports are checked against the references in the repo.
#
#   git clone https://github.com/PS12007/kernelport && cd kernelport && bash scripts/amd_check.sh
#
# Writes amd_results.txt; send that file back. Untested until its first run on AMD.
set -uo pipefail

OUT=amd_results.txt
exec > >(tee "$OUT") 2>&1

echo "== machine"
rocm-smi --showproductname 2>/dev/null || true
cat /opt/rocm/.info/version 2>/dev/null || true
python3 --version

echo "== python env (reuses the image's ROCm torch if it has one)"
python3 -m venv --system-site-packages .venv-amd
source .venv-amd/bin/activate
python -m pip install -q --upgrade pip

if ! python -c "import torch, sys; sys.exit(0 if torch.version.hip and torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "-- installing ROCm torch from ${ROCM_TORCH_INDEX:=https://download.pytorch.org/whl/rocm6.4}"
    python -m pip install -q torch --index-url "$ROCM_TORCH_INDEX"
fi
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.get_device_name(0))"

if ! python -c "import cupy, sys; sys.exit(0 if cupy.cuda.runtime.is_hip else 1)" 2>/dev/null; then
    echo "-- installing ROCm CuPy (provides hipRTC)"
    ok=0
    for wheel in cupy-rocm-6-0 cupy-rocm-5-0; do
        if python -m pip install -q "$wheel"; then ok=1; break; fi
    done
    if [ "$ok" = 0 ] || ! python -c "import cupy; cupy.zeros(1)" 2>/dev/null; then
        echo "-- no working CuPy wheel; building CuPy for ROCm from source (10-30 min)"
        python -m pip uninstall -y -q cupy-rocm-6-0 cupy-rocm-5-0 2>/dev/null || true
        GFX=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*')
        CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET="${GFX:-gfx942}" \
            python -m pip install -q --no-binary cupy cupy
    fi
fi
python -c "import cupy; print('cupy', cupy.__version__, 'is_hip', cupy.cuda.runtime.is_hip)"

python -m pip install -q -e ".[dev]" numba

echo "== kernelport doctor"
kernelport doctor

echo "== zoo: hip ports"
kernelport zoo --backend hip
echo "== zoo: triton ports on ROCm"
kernelport zoo --backend triton

echo "== llm.c example: triton ports on ROCm"
(cd examples/llm.c && kernelport verify --backend triton && kernelport status)

echo "== tests (AMD-specific ones run here, CUDA-only ones skip)"
python -m pytest -q -rs tests/test_hip_backend.py tests/test_backends.py

echo "== done; results in $OUT"
