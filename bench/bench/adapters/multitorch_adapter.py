"""multitorch adapter: wraps calcXAS_cached / _from_scratch / _batch / calcRIXS.

The adapter class is parameterized by "mode" which selects between the
three multitorch execution paths we benchmark separately:

  mode="cached"      → preload_fixture + calcXAS_cached   (realistic usage)
  mode="from_scratch"→ calcXAS_from_scratch                (apples-to-apples vs Fortran)
  mode="batch"       → preload_fixture + calcXAS_batch     (Phase 2 sweep mode)

calcXAS_from_scratch is Oh-only and single-configuration (no charge
transfer). We report NotSupported for fixtures it can't handle.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from bench.adapters.base import Adapter, AdapterResult, NotSupported
from bench.config import BenchCell, FIXTURE_BY_NAME


class MultitorchAdapter(Adapter):
    supports_cuda = True
    supports_batch = True

    def __init__(self, mode: str = "cached"):
        if mode not in ("cached", "from_scratch", "batch"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.name = f"multitorch_{mode}"
        self._fixture_cache: Dict[str, object] = {}  # fixture_name → CachedFixture
        self._imports_done = False

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def cold_start(self) -> None:
        import multitorch  # noqa: F401  — triggers import cost timing
        import torch  # noqa: F401
        self._imports_done = True

    def warm(self, cell: BenchCell) -> None:
        # One untimed call through the same path run() will take.
        self.run(cell)

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_cached_fixture(self, fixture_name: str):
        from multitorch import preload_fixture
        if fixture_name not in self._fixture_cache:
            fx = FIXTURE_BY_NAME[fixture_name]
            self._fixture_cache[fixture_name] = preload_fixture(
                fx.element, fx.valence, fx.sym
            )
        return self._fixture_cache[fixture_name]

    @staticmethod
    def _to_numpy(x, y):
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().astype(np.float64)
        else:
            x = np.asarray(x, dtype=np.float64)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().astype(np.float64)
        else:
            y = np.asarray(y, dtype=np.float64)
        return x, y

    # ─────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────

    def run(self, cell: BenchCell) -> AdapterResult:
        if cell.calctype != "xas":
            raise NotSupported(
                f"multitorch_{self.mode} only benchmarks XAS for now; "
                f"RIXS/XES support comes in the full-matrix milestone"
            )

        fx = FIXTURE_BY_NAME[cell.fixture]
        slater = cell.slater if cell.slater is not None else 0.8
        soc = cell.soc if cell.soc is not None else 1.0

        if self.mode == "cached":
            return self._run_cached(cell, fx, slater, soc)
        if self.mode == "from_scratch":
            return self._run_from_scratch(cell, fx, slater, soc)
        if self.mode == "batch":
            return self._run_batch(cell, fx, slater, soc)
        raise RuntimeError("unreachable")

    # ─── mode: cached ────────────────────────────────────────

    def _run_cached(self, cell, fx, slater, soc) -> AdapterResult:
        from multitorch import calcXAS_cached
        cache = self._get_cached_fixture(cell.fixture)

        if cell.batch_size == 1:
            x, y = calcXAS_cached(
                cache, slater=slater, soc=soc,
                nbins=cell.nbins, T=cell.T, device=cell.device,
            )
            x, y = self._to_numpy(x, y)
            return AdapterResult(x=x, y=y, meta={"pseudo_batch": False})
        else:
            # Emulate batch by serial calls (matches Fortran emulation).
            xs, ys = [], []
            for _ in range(cell.batch_size):
                xi, yi = calcXAS_cached(
                    cache, slater=slater, soc=soc,
                    nbins=cell.nbins, T=cell.T, device=cell.device,
                )
                xs.append(xi)
                ys.append(yi)
            # Return the last spectrum for parity checking; the relevant
            # number here is the total batch time, not per-spectrum parity.
            x, y = self._to_numpy(xs[-1], ys[-1])
            return AdapterResult(x=x, y=y, meta={"pseudo_batch": True})

    # ─── mode: from_scratch ──────────────────────────────────

    def _run_from_scratch(self, cell, fx, slater, soc) -> AdapterResult:
        if fx.sym != "oh":
            raise NotSupported(
                f"calcXAS_from_scratch only supports Oh, not {fx.sym!r}"
            )
        if fx.has_charge_transfer:
            raise NotSupported(
                "calcXAS_from_scratch does not support charge transfer"
            )
        from multitorch import calcXAS_from_scratch
        cf = {"tendq": 1.0}

        if cell.batch_size == 1:
            x, y = calcXAS_from_scratch(
                element=fx.element, valence=fx.valence, cf=cf,
                slater=slater, soc=soc,
                nbins=cell.nbins, T=cell.T, device=cell.device,
            )
            x, y = self._to_numpy(x, y)
            return AdapterResult(x=x, y=y, meta={"pseudo_batch": False})
        xs, ys = [], []
        for _ in range(cell.batch_size):
            xi, yi = calcXAS_from_scratch(
                element=fx.element, valence=fx.valence, cf=cf,
                slater=slater, soc=soc,
                nbins=cell.nbins, T=cell.T, device=cell.device,
            )
            xs.append(xi)
            ys.append(yi)
        x, y = self._to_numpy(xs[-1], ys[-1])
        return AdapterResult(x=x, y=y, meta={"pseudo_batch": True})

    # ─── mode: batch ─────────────────────────────────────────

    def _run_batch(self, cell, fx, slater, soc) -> AdapterResult:
        if cell.batch_size < 2:
            raise NotSupported("multitorch_batch only applies for batch_size >= 2")
        import torch
        from multitorch import calcXAS_batch
        cache = self._get_cached_fixture(cell.fixture)

        N = cell.batch_size
        slater_values = torch.full((N,), float(slater), dtype=torch.float64)
        soc_values = torch.full((N,), float(soc), dtype=torch.float64)
        y_batch = calcXAS_batch(
            cache,
            slater_values=slater_values, soc_values=soc_values,
            nbins=cell.nbins, T=cell.T, device=cell.device,
        )
        # Return the last spectrum for parity; record the batch shape.
        # Reconstruct x-grid — calcXAS_batch computes a shared grid but
        # doesn't return it, so we derive it from a single cached call.
        x_grid, _ = self._single_grid(cache, slater, soc, cell)
        x_np = x_grid.detach().cpu().numpy().astype(np.float64)
        y_np = y_batch[-1].detach().cpu().numpy().astype(np.float64)
        # Note: the batch's shared x may differ from the single-call x;
        # we re-broaden the returned y on its native shared grid elsewhere
        # (the parity module handles mismatched x-grids via interpolation).
        return AdapterResult(
            x=x_np, y=y_np,
            meta={"pseudo_batch": False, "batch_shape": list(y_batch.shape)},
        )

    def _single_grid(self, cache, slater, soc, cell):
        """Utility: obtain x grid from one cached call (not timed)."""
        from multitorch import calcXAS_cached
        return calcXAS_cached(
            cache, slater=slater, soc=soc,
            nbins=cell.nbins, T=cell.T, device=cell.device,
        )
