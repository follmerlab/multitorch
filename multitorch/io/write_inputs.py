"""
Input file generators for the ttmult/ttrcg/ttban Fortran suite.

Ported verbatim from pyctm/pyctm/write_RCG.py, write_RAC.py, write_BAN.py.
This module contains only string I/O logic — no physics — and is used to
bootstrap calculations until the full PyTorch physics layers (Phases 2-4)
are complete.

Key functions:
  write_rcg(filename, element, trans, slater, soc, eground, efinal)
  write_rac(filename, sym, edge, cf, nconf)
  write_ban(filename, sym, cf, delta, u, lmct, mlct, calctype)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# RCG file writer
# ─────────────────────────────────────────────────────────────

def write_rcg(filename: str, element: str, trans: dict, slater: float,
              soc: float, eground: float = 0.0, efinal: float = 20.0,
              **kwargs) -> None:
    """
    Generate a .rcg input file for ttrcg.

    Thin wrapper — calls write_ct_rcg if multiple configurations are given,
    or write_cf_rcg for a single crystal-field configuration.

    Parameters
    ----------
    filename : str
        Base filename (without extension); creates <filename>.rcg
    element : str
        Element symbol, e.g. 'Ni', 'Fe'
    trans : dict
        Transition definitions. Keys: 'ion', and optionally 'lmct'/'mlct'.
    slater : float
        Slater integral reduction factor (0-1, typical 0.8)
    soc : float
        SOC reduction factor (0-1, typical 1.0)
    eground : float
        Ground-state energy offset (Ry), usually 0.0
    efinal : float
        Final-state energy offset (Ry), usually 20.0 for XAS
    """
    # Import original pyctm writers as fallback during development
    # Once the full port is complete, replace with native implementations
    _ensure_pyctm_path()
    from pyctm.write_RCG import writeRCG as _writeRCG
    _writeRCG(filename, element, trans, slater, soc, eground, efinal, **kwargs)


def write_rac(filename: str, sym: str, edge: str, cf: dict,
              nconf: int = 1) -> None:
    """Generate a .rac input file for ttrac."""
    _ensure_pyctm_path()
    from pyctm.write_RAC import writeRAC as _writeRAC
    _writeRAC(filename, sym, edge, cf, nconf)


def write_ban(filename: str, sym: str, cf: dict,
              delta: Optional[dict] = None, u: Optional[list] = None,
              lmct: Optional[dict] = None, mlct: Optional[dict] = None,
              calctype: str = "xas", nband: Optional[int] = None,
              bandwidth: Optional[float] = None) -> None:
    """Generate a .ban input file for ttban."""
    _ensure_pyctm_path()
    from pyctm.write_BAN import writeBAN as _writeBAN
    _writeBAN(filename, sym, cf, delta, u, lmct, mlct, calctype, nband, bandwidth)


def _ensure_pyctm_path() -> None:
    """Add pyctm to sys.path if not already importable."""
    try:
        import pyctm  # noqa: F401
    except ImportError:
        # Add sibling pyctm package to path
        here = Path(__file__).resolve().parent
        root = here.parent.parent.parent  # multitorch root
        pyctm_path = root.parent / "pyctm"
        if pyctm_path.exists() and str(pyctm_path) not in sys.path:
            sys.path.insert(0, str(pyctm_path))
