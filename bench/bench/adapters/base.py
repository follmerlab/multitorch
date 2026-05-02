"""Base class that defines the contract every adapter implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from bench.config import BenchCell


@dataclass
class AdapterResult:
    """Output of a single adapter.run() call.

    x / y are numpy float64 arrays on a shared length. meta carries
    impl-specific breakdown timings (subprocess_s, fileio_s) that the
    harness collects into the JSON record.
    """

    x: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


class Adapter(ABC):
    """Contract every benchmarked implementation must satisfy.

    Lifecycle:

        a = MyAdapter()
        a.cold_start()          # timed once per session
        a.warm(cell)            # untimed throwaway run
        for _ in range(reps):
            with wall_timer() as t:
                out = a.run(cell)    # (x, y, meta)

    An adapter may raise :class:`NotSupported` to signal a cell it
    cannot execute (e.g. ttmult_raw asked for cuda). The harness
    records `status="skipped"` in that case.
    """

    name: str = "base"
    supports_cuda: bool = False
    supports_batch: bool = False

    @abstractmethod
    def cold_start(self) -> None:
        """One-time initialization: imports, binary-path resolution, etc."""

    @abstractmethod
    def warm(self, cell: BenchCell) -> None:
        """One untimed throwaway run to warm caches / jit / CUDA kernels."""

    @abstractmethod
    def run(self, cell: BenchCell) -> AdapterResult:
        """Execute the cell and return spectrum + breakdown timings."""

    def supports(self, cell: BenchCell) -> bool:
        """Return True if this adapter can handle the cell."""
        if cell.device == "cuda" and not self.supports_cuda:
            return False
        if cell.batch_size > 1 and not self.supports_batch:
            # Fortran impls emulate batch via serial loops — they
            # "support" batch in the pseudo_batch sense. The concrete
            # adapter classes override this check when needed.
            return True
        return True


class NotSupported(RuntimeError):
    """Raise from adapter.run() to signal 'this cell is not applicable'."""
