"""pyctm adapter: wraps pyctm.calc.calcXAS in a per-call tempdir.

pyctm is the legacy Python orchestrator that writes .rcg/.rac/.ban
input files, invokes the Fortran binaries via pyttmult, reads the
.oba output, and broadens to produce (x, y). Each call is a full
subprocess pipeline, so batch_size=N means N sequential calls
(flagged pseudo_batch=true).

Requires the $ttmult environment variable to point at the top of the
ttmult directory (expected default: $WORKSPACE_ROOT/ttmult/). Fails
fast in cold_start() if the binaries aren't found.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from bench.adapters.base import Adapter, AdapterResult, NotSupported
from bench.config import BenchCell, FIXTURE_BY_NAME, PYCTM_ROOT, TTMULT_ROOT


@contextmanager
def _cwd(path: Path):
    """Context manager to cd into path and guarantee restoration."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


class PyctmAdapter(Adapter):
    supports_cuda = False
    supports_batch = False        # pseudo-batches via serial loops

    name = "pyctm"

    def __init__(self):
        self._imports_done = False
        self._bin_root: Optional[Path] = None

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def cold_start(self) -> None:
        # Make pyctm + pyttmult importable.
        if str(PYCTM_ROOT) not in sys.path:
            sys.path.insert(0, str(PYCTM_ROOT))
        pyttmult_root = PYCTM_ROOT.parent / "pyttmult"
        if str(pyttmult_root) not in sys.path:
            sys.path.insert(0, str(pyttmult_root))

        # pyttmult reads $ttmult via os.getenv, so set it if unset.
        if "ttmult" not in os.environ:
            os.environ["ttmult"] = str(TTMULT_ROOT)
        self._bin_root = Path(os.environ["ttmult"]) / "bin"

        # Verify required binaries exist.
        required = ["ttrcg", "ttrac", "ttban_exact", "rcn31", "rcn2"]
        missing = [b for b in required if not (self._bin_root / b).exists()]
        if missing:
            raise FileNotFoundError(
                f"Fortran binaries not found at {self._bin_root}: {missing}. "
                f"Set $ttmult to the ttmult root, or install the binaries."
            )

        # Trigger the first pyctm import cost explicitly.
        import pyctm  # noqa: F401
        self._imports_done = True

    def warm(self, cell: BenchCell) -> None:
        self.run(cell)

    # ─────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────

    def run(self, cell: BenchCell) -> AdapterResult:
        if cell.calctype != "xas":
            raise NotSupported(
                f"pyctm adapter currently benchmarks XAS only; "
                f"RIXS/XES come later in the matrix"
            )

        fx = FIXTURE_BY_NAME[cell.fixture]
        if fx.has_charge_transfer:
            raise NotSupported(
                "pyctm adapter skips charge-transfer fixtures in the smoke "
                "matrix — needs explicit delta/lmct/mlct kwargs to work"
            )

        # pyctm calls writeXAS → runXAS → getXAS.
        from pyctm.calc import calcXAS

        slater = cell.slater if cell.slater is not None else 0.8
        soc = cell.soc if cell.soc is not None else 1.0

        # Track total subprocess time by wrapping pyttmult.ttmult — done
        # as a monkeypatch-style hook. For the smoke adapter we treat the
        # full calcXAS call as one black-box Fortran window. A more
        # granular decomposition is added in the pyttmult_adapter.
        last_x = None
        last_y = None
        iters = cell.batch_size if cell.batch_size > 1 else 1

        with tempfile.TemporaryDirectory(prefix="bench_pyctm_") as tmp:
            tmp_path = Path(tmp)
            with _cwd(tmp_path):
                for i in range(iters):
                    stub = f"run_{i}"
                    x, y = calcXAS(
                        filename=stub,
                        element=fx.element,
                        valence=fx.valence,
                        sym=fx.sym,
                        edge=fx.edge,
                        cf={"tendq": 1.0},
                        slater=slater,
                        soc=soc,
                        save=False, get=True, run=True, verbose=False,
                    )
                    last_x = np.asarray(x, dtype=np.float64)
                    last_y = np.asarray(y, dtype=np.float64)

        if last_x is None or last_y is None:
            raise RuntimeError("pyctm returned no spectrum")

        return AdapterResult(
            x=last_x, y=last_y,
            meta={"pseudo_batch": iters > 1, "iters": iters},
        )
