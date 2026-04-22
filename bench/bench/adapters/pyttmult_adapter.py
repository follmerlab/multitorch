"""pyttmult adapter: times pyttmult.ttmult() in isolation from pyctm wrapping.

Flow:

  1. pyctm.write.writeXAS(autorun=False) generates the Fortran input
     files (.rcg, .rac, .rcn, rcn31.out, etc.) — this is the setup
     cost, timed as ``setup_s``.

  2. pyttmult.ttmult(filename, calctype, bandertype='exact') runs the
     Fortran pipeline: ttrcg + ttrac for non-CT, ttrcg + ttrac + ttban
     for CT. This window is timed as ``fortran_s`` and is the primary
     measurement of interest for this adapter.

  3. pyctm.plot.getXAS(filename) reads the .ora (non-CT) or .oba (CT)
     file produced by the Fortran stack and broadens it into (x, y).
     Timed as ``broaden_s``.

Compared to pyctm_adapter (which runs the full pyctm.calc.calcXAS
pipeline in one opaque call), pyttmult_adapter exposes the three-
stage decomposition the plan calls for — so the difference
``pyctm − pyttmult`` isolates pyctm's orchestration overhead.
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from bench.adapters.base import Adapter, AdapterResult, NotSupported
from bench.config import BenchCell, FIXTURE_BY_NAME, PYCTM_ROOT, PYTTMULT_ROOT, TTMULT_ROOT
from bench.harness import wall_timer


@contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def _cf_for(fx) -> dict:
    """Build a crystal-field dict pyctm.writeRAC will accept for this symmetry."""
    if fx.sym == "oh":
        return {"tendq": 1.0}
    # d4h needs dt and ds; default them to a small splitting.
    return {"tendq": 1.0, "dt": 0.0, "ds": 0.0}


class PyttmultAdapter(Adapter):
    supports_cuda = False
    supports_batch = False        # pseudo-batches via serial calls

    name = "pyttmult"

    def __init__(self):
        self._imports_done = False
        self._bin_root: Optional[Path] = None

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def cold_start(self) -> None:
        if str(PYCTM_ROOT) not in sys.path:
            sys.path.insert(0, str(PYCTM_ROOT))
        if str(PYTTMULT_ROOT) not in sys.path:
            sys.path.insert(0, str(PYTTMULT_ROOT))

        if "ttmult" not in os.environ:
            os.environ["ttmult"] = str(TTMULT_ROOT)
        self._bin_root = Path(os.environ["ttmult"]) / "bin"
        required = ["ttrcg", "ttrac", "ttban_exact", "rcn31", "rcn2"]
        missing = [b for b in required if not (self._bin_root / b).exists()]
        if missing:
            raise FileNotFoundError(
                f"Fortran binaries not found at {self._bin_root}: {missing}"
            )

        import pyttmult  # noqa: F401
        import pyctm     # noqa: F401
        self._imports_done = True

    def warm(self, cell: BenchCell) -> None:
        self.run(cell)

    # ─────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────

    def run(self, cell: BenchCell) -> AdapterResult:
        if cell.calctype != "xas":
            raise NotSupported("pyttmult adapter currently benchmarks XAS only")

        fx = FIXTURE_BY_NAME[cell.fixture]
        if fx.has_charge_transfer:
            raise NotSupported(
                "pyttmult adapter skips CT fixtures in the smoke matrix — "
                "needs explicit delta/lmct/mlct kwargs"
            )

        from pyctm.write import writeXAS
        from pyctm.plot import getXAS
        from pyttmult.ttmult import ttmult as run_ttmult

        slater = cell.slater if cell.slater is not None else 0.8
        soc = cell.soc if cell.soc is not None else 1.0
        cf = _cf_for(fx)

        iters = cell.batch_size if cell.batch_size > 1 else 1
        calctype_cmd = "all" if fx.has_charge_transfer else "rcg_rac"

        last_x = None
        last_y = None
        setup_accum = 0.0
        fortran_accum = 0.0
        broaden_accum = 0.0

        with tempfile.TemporaryDirectory(prefix="bench_pyttmult_") as tmp:
            tmp_path = Path(tmp)
            with _cwd(tmp_path):
                for i in range(iters):
                    stub = f"run_{i}"

                    with wall_timer() as t_setup:
                        writeXAS(
                            stub,
                            element=fx.element, valence=fx.valence,
                            sym=fx.sym, edge=fx.edge, cf=cf,
                            slater=slater, soc=soc,
                            autorun=False,
                        )
                    setup_accum += t_setup.elapsed

                    with wall_timer() as t_fortran:
                        run_ttmult(stub, calctype=calctype_cmd,
                                   bandertype="exact", verbose=False)
                    fortran_accum += t_fortran.elapsed

                    with wall_timer() as t_broaden:
                        x, y = getXAS(stub)
                    broaden_accum += t_broaden.elapsed

                    last_x = np.asarray(x, dtype=np.float64)
                    last_y = np.asarray(y, dtype=np.float64)

        if last_x is None or last_y is None:
            raise RuntimeError("pyttmult returned no spectrum")

        return AdapterResult(
            x=last_x, y=last_y,
            meta={
                "pseudo_batch": iters > 1,
                "iters": iters,
                "setup_s": setup_accum,
                "fortran_s": fortran_accum,
                "broaden_s": broaden_accum,
                "calctype_cmd": calctype_cmd,
            },
        )
