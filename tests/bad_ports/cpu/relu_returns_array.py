"""BAD cpu port: returns a new array instead of writing the output in place."""

import numpy as np


def run(x, y, n):
    return np.maximum(x, 0)
