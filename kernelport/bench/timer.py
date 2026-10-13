"""Device-synchronised kernel timing.

Warm up, then time each launch individually with device events and report
the median. On NVIDIA and ROCm we use torch.cuda.Event (ROCm maps it to HIP
events). CPU-side timers are only used for backends with no device events,
and those numbers are labelled as not comparable.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


@dataclass
class Timing:
    """Timing summary in milliseconds."""

    median_ms: float
    min_ms: float
    max_ms: float
    iters: int
    method: str  # "cuda_event" or "wall_clock"

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return asdict(self)


def time_cuda(launch: Callable[[], None], warmup: int = DEFAULT_WARMUP,
              iters: int = DEFAULT_ITERS) -> Timing:
    """Median time of ``launch()`` measured with CUDA/HIP events."""
    import torch

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        launch()
        e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return Timing(statistics.median(times), min(times), max(times), iters, "cuda_event")


def time_wall(launch: Callable[[], None], sync: Callable[[], None] = lambda: None,
              warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS) -> Timing:
    """Median wall-clock time of ``launch(); sync()``."""
    for _ in range(warmup):
        launch()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return Timing(statistics.median(times), min(times), max(times), iters, "wall_clock")
