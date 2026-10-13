"""BAD cpu port: rebinds the local name y, so the caller's output array is untouched."""

import numpy as np


def run(x, y, n):
    y = np.maximum(x, 0)  # noqa: F841  (should be y[:] = ...)
