"""Timing primitives: wall-clock, CPU, CUDA event, rep statistics.

All timer context managers expose the elapsed value in seconds (float)
as ``timer.elapsed`` after the ``with`` block closes. CUDA timing uses
explicit synchronization and torch.cuda.Event pairs because
``time.perf_counter`` on a CUDA workload will return before the kernel
actually finishes.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class _Timer:
    """Context-manager style wall-clock + CPU-time timer (nanoseconds → seconds)."""

    elapsed: float = 0.0           # wall-clock seconds
    cpu_elapsed: float = 0.0       # sum of user + system CPU seconds
    _t0_wall: int = field(default=0, init=False, repr=False)
    _t0_cpu: int = field(default=0, init=False, repr=False)

    def __enter__(self) -> "_Timer":
        self._t0_wall = time.perf_counter_ns()
        self._t0_cpu = time.process_time_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = (time.perf_counter_ns() - self._t0_wall) / 1e9
        self.cpu_elapsed = (time.process_time_ns() - self._t0_cpu) / 1e9


def wall_timer() -> _Timer:
    return _Timer()


@contextmanager
def cuda_event_timer(device: Optional[str] = None):
    """Yield a dict that gets populated with {'gpu_ms': float} on exit.

    Requires CUDA to be available. Uses torch.cuda.Event for authoritative
    GPU-side timing (the kernel queue may not drain by the time CPU
    ``perf_counter`` stops, so we also torch.cuda.synchronize to bracket).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_event_timer called without CUDA available")

    result: dict = {"gpu_ms": 0.0}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    start.record()
    try:
        yield result
    finally:
        end.record()
        torch.cuda.synchronize(device)
        result["gpu_ms"] = start.elapsed_time(end)


# ─────────────────────────────────────────────────────────────
# Repetition statistics
# ─────────────────────────────────────────────────────────────


@dataclass
class RepStats:
    samples: List[float]

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def median(self) -> float:
        import statistics
        return float(statistics.median(self.samples))

    @property
    def mean(self) -> float:
        return float(sum(self.samples) / max(self.n, 1))

    @property
    def std(self) -> float:
        import statistics
        return float(statistics.pstdev(self.samples)) if self.n > 1 else 0.0

    @property
    def iqr(self) -> float:
        import statistics
        if self.n < 2:
            return 0.0
        q = statistics.quantiles(self.samples, n=4)
        return float(q[2] - q[0])

    @property
    def q25(self) -> float:
        import statistics
        if self.n < 2:
            return float(self.samples[0]) if self.samples else 0.0
        return float(statistics.quantiles(self.samples, n=4)[0])

    @property
    def q75(self) -> float:
        import statistics
        if self.n < 2:
            return float(self.samples[0]) if self.samples else 0.0
        return float(statistics.quantiles(self.samples, n=4)[2])

    @property
    def min(self) -> float:
        return float(min(self.samples)) if self.samples else 0.0

    @property
    def max(self) -> float:
        return float(max(self.samples)) if self.samples else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "iqr": self.iqr,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
            "samples": list(self.samples),
        }
