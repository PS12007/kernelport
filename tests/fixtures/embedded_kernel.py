"""Fixture: a CUDA kernel embedded in Python, as CuPy users write it."""

import cupy as cp  # noqa: F401  (never imported by tests)

SCALE_SRC = r"""
extern "C" __global__
void scale(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i];
}
"""

kernel = None  # cp.RawKernel(SCALE_SRC, "scale")
