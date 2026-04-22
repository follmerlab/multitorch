"""ttmult_raw adapter: direct subprocess invocation of ttrcg/ttrac/ttban_exact.

Bypasses the pyttmult wrapper so we can time each Fortran stage in
isolation. The file-dance (fort.10 linking, fort.14/15/50/72/73/74
shuffling) replicates what pyttmult.ttmult.runrcg / runrac / runban do,
just with subprocess.run instead of os.system so stdout/stderr capture
is tighter and the Python-side overhead is minimal.

This adapter establishes the pure-Fortran floor: ``pyttmult − ttmult_raw``
gives the pyttmult Python wrapper overhead (shell fork, etc.) and
``pyctm − pyttmult`` gives the pyctm orchestration overhead.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

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
    if fx.sym == "oh":
        return {"tendq": 1.0}
    return {"tendq": 1.0, "dt": 0.0, "ds": 0.0}


def _run_binary(binary: Path, *, stdin_file: Optional[Path] = None,
                stdout_file: Optional[Path] = None,
                log_file: Optional[Path] = None,
                cwd: Optional[Path] = None) -> float:
    """Run one Fortran binary; return elapsed seconds. Raises on nonzero exit."""
    stdin_fh = open(stdin_file, "r") if stdin_file is not None else subprocess.DEVNULL
    log_fh = open(log_file, "w") if log_file is not None else subprocess.DEVNULL
    stdout_fh = open(stdout_file, "w") if stdout_file is not None else log_fh
    try:
        with wall_timer() as t:
            proc = subprocess.run(
                [str(binary)],
                stdin=stdin_fh,
                stdout=stdout_fh,
                stderr=log_fh,
                cwd=str(cwd) if cwd is not None else None,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{binary.name} exited with code {proc.returncode}; "
                f"see {log_file}"
            )
        return t.elapsed
    finally:
        if stdin_file is not None:
            stdin_fh.close()
        if log_file is not None:
            log_fh.close()
        if stdout_file is not None and stdout_file != log_file:
            stdout_fh.close()


# ─────────────────────────────────────────────────────────────
# Fortran stage runners (replicate pyttmult's file dance)
# ─────────────────────────────────────────────────────────────


def _run_ttrcg(stub: str, bin_root: Path, cwd: Path) -> float:
    """ttrcg reads <stub>.rcg (via fort.10) + CFP tables; produces .m14."""
    # Stage inputs
    if (cwd / "fort.10").exists():
        (cwd / "fort.10").unlink()
    os.link(str(cwd / f"{stub}.rcg"), str(cwd / "fort.10"))
    for n in (72, 73, 74):
        fort = cwd / f"fort.{n}"
        if not fort.exists():
            shutil.copy(str(bin_root / f"rcg_cfp{n}"), str(fort))

    # Run
    t = _run_binary(bin_root / "ttrcg", log_file=cwd / "ttrcg.log", cwd=cwd)

    # Stage outputs
    shutil.move(str(cwd / "fort.14"), str(cwd / f"{stub}.m14"))
    if (cwd / "fort.9").exists():
        shutil.move(str(cwd / "fort.9"), str(cwd / f"{stub}.org"))
    if (cwd / "fort.10").exists():
        (cwd / "fort.10").unlink()
    return t


def _run_ttrac(stub: str, bin_root: Path, cwd: Path) -> float:
    """ttrac reads <stub>.rac on stdin, <stub>.m14 + <stub>.ora on argv."""
    # ttrac is invoked with args: ttrac <stub>.m14 <stub>.ora  < <stub>.rac
    # subprocess.run does not expand the '<' redirect, so we manage stdin ourselves.
    stdin_fh = open(cwd / f"{stub}.rac", "r")
    log_fh = open(cwd / "ttrac.log", "w")
    try:
        with wall_timer() as t:
            proc = subprocess.run(
                [str(bin_root / "ttrac"), f"{stub}.m14", f"{stub}.ora"],
                stdin=stdin_fh,
                stdout=log_fh,
                stderr=log_fh,
                cwd=str(cwd),
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ttrac exited with code {proc.returncode}; see ttrac.log"
            )
    finally:
        stdin_fh.close()
        log_fh.close()
    if (cwd / "rme_out.dat").exists():
        shutil.move(str(cwd / "rme_out.dat"), str(cwd / f"{stub}.m15"))
    return t.elapsed


def _run_ttban_exact(stub: str, bin_root: Path, cwd: Path) -> float:
    """ttban_exact reads FTN14/FTN15/fort.50; produces fort.44 → .oba."""
    if (cwd / f"{stub}.m14").exists():
        shutil.move(str(cwd / f"{stub}.m14"), str(cwd / "FTN14"))
    if (cwd / f"{stub}.m15").exists():
        shutil.move(str(cwd / f"{stub}.m15"), str(cwd / "FTN15"))
    if (cwd / "fort.50").exists():
        (cwd / "fort.50").unlink()
    os.link(str(cwd / f"{stub}.ban"), str(cwd / "fort.50"))

    t = _run_binary(bin_root / "ttban_exact",
                    log_file=cwd / "ttban_exact.log", cwd=cwd)

    # Restore intermediate files.
    if (cwd / "FTN14").exists():
        shutil.move(str(cwd / "FTN14"), str(cwd / f"{stub}.m14"))
    if (cwd / "FTN15").exists():
        shutil.move(str(cwd / "FTN15"), str(cwd / f"{stub}.m15"))
    if (cwd / "fort.44").exists():
        shutil.move(str(cwd / "fort.44"), str(cwd / f"{stub}.oba"))
    for junk in ("FTN98", "FTN99", "fort.50", "fort.43"):
        p = cwd / junk
        if p.exists():
            p.unlink()
    return t


# ─────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────


class TtmultRawAdapter(Adapter):
    supports_cuda = False
    supports_batch = False

    name = "ttmult_raw"

    def __init__(self):
        self._imports_done = False
        self._bin_root: Optional[Path] = None

    # Lifecycle
    def cold_start(self) -> None:
        if str(PYCTM_ROOT) not in sys.path:
            sys.path.insert(0, str(PYCTM_ROOT))
        if str(PYTTMULT_ROOT) not in sys.path:
            sys.path.insert(0, str(PYTTMULT_ROOT))
        if "ttmult" not in os.environ:
            os.environ["ttmult"] = str(TTMULT_ROOT)
        self._bin_root = Path(os.environ["ttmult"]) / "bin"
        required = ["ttrcg", "ttrac", "ttban_exact"]
        missing = [b for b in required if not (self._bin_root / b).exists()]
        if missing:
            raise FileNotFoundError(
                f"Fortran binaries not found at {self._bin_root}: {missing}"
            )
        # We reuse pyctm's file writers and .ora/.oba broadening.
        import pyctm  # noqa: F401
        self._imports_done = True

    def warm(self, cell: BenchCell) -> None:
        self.run(cell)

    def run(self, cell: BenchCell) -> AdapterResult:
        if cell.calctype != "xas":
            raise NotSupported("ttmult_raw adapter currently benchmarks XAS only")

        fx = FIXTURE_BY_NAME[cell.fixture]
        if fx.has_charge_transfer:
            raise NotSupported("ttmult_raw skips CT fixtures in smoke")

        from pyctm.write import writeXAS
        from pyctm.plot import getXAS

        slater = cell.slater if cell.slater is not None else 0.8
        soc = cell.soc if cell.soc is not None else 1.0
        cf = _cf_for(fx)

        iters = cell.batch_size if cell.batch_size > 1 else 1
        last_x = None
        last_y = None
        setup_accum = 0.0
        ttrcg_accum = 0.0
        ttrac_accum = 0.0
        ttban_accum = 0.0
        broaden_accum = 0.0

        with tempfile.TemporaryDirectory(prefix="bench_ttmult_raw_") as tmp:
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

                    # Fortran stages, each timed independently.
                    ttrcg_accum += _run_ttrcg(stub, self._bin_root, tmp_path)
                    ttrac_accum += _run_ttrac(stub, self._bin_root, tmp_path)
                    if fx.has_charge_transfer:
                        ttban_accum += _run_ttban_exact(stub, self._bin_root, tmp_path)

                    with wall_timer() as t_broaden:
                        x, y = getXAS(stub)
                    broaden_accum += t_broaden.elapsed

                    last_x = np.asarray(x, dtype=np.float64)
                    last_y = np.asarray(y, dtype=np.float64)

        if last_x is None or last_y is None:
            raise RuntimeError("ttmult_raw returned no spectrum")

        return AdapterResult(
            x=last_x, y=last_y,
            meta={
                "pseudo_batch": iters > 1,
                "iters": iters,
                "setup_s": setup_accum,
                "ttrcg_s": ttrcg_accum,
                "ttrac_s": ttrac_accum,
                "ttban_s": ttban_accum,
                "fortran_s": ttrcg_accum + ttrac_accum + ttban_accum,
                "broaden_s": broaden_accum,
            },
        )
