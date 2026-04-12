"""
Crystal field Hamiltonian construction.

Encodes the Butler-Wybourne branch coefficients for the crystal field
operators in Oh, D4h, and C4h symmetries. These coefficients appear in
the `.rac` file as BRANCH entries and must be reproduced EXACTLY (they
are exact algebraic values, not approximations).

From pyctm/write_RAC.py and Butler-Wybourne group theory tables:

D4h symmetry:
  X40 = 6*sqrt(30) * (10Dq / 10)   → 4-th rank irrep strength
  X420 = -5/2 * sqrt(42) * dt       → mixed rank-4/2 dt parameter
  X220 = -sqrt(70) * ds              → rank-2 ds parameter

Oh symmetry:
  X40 = 6*sqrt(30) * (10Dq / 10)   → same as D4h, just one parameter

Physical meaning:
  10Dq: cubic crystal field splitting (in eV)
  Dt: tetragonal distortion (rank-4 axial, eV)
  Ds: tetragonal distortion (rank-2 axial, eV)
"""
from __future__ import annotations
from typing import Dict
import math


# Pre-computed exact algebraic crystal field coefficients
# These are from Butler-Wybourne tables (write_RAC.py shellDict/hamDict)
_CF_BRANCH_COEFFS = {
    "oh": {
        "X40": 6.0 * math.sqrt(30.0),          # coefficient × 10Dq/10
    },
    "d4h": {
        "X40": 6.0 * math.sqrt(30.0),          # × 10Dq/10
        "X420": -2.5 * math.sqrt(42.0),         # × dt
        "X220": -math.sqrt(70.0),               # × ds
    },
    "c4h": {
        "X40": 6.0 * math.sqrt(30.0),
        "X420": -2.5 * math.sqrt(42.0),
        "X220": -math.sqrt(70.0),
    },
}


def get_cf_branch_values(
    sym: str,
    tendq: float = 0.0,
    ds: float = 0.0,
    dt: float = 0.0,
) -> Dict[str, float]:
    """
    Compute crystal field branch coefficient × parameter values.

    These are the values that multiply the RME SHELL operator blocks
    in the Hamiltonian assembly.

    Parameters
    ----------
    sym : str
        Symmetry: 'oh', 'd4h', or 'c4h'.
    tendq : float
        10Dq crystal field parameter (eV). (Note: ban uses 10Dq - 35dt/6)
    ds : float
        Ds tetragonal distortion parameter (eV).
    dt : float
        Dt tetragonal distortion parameter (eV).

    Returns
    -------
    dict mapping branch name → numeric value (ready to multiply RME blocks)
    """
    sym = sym.lower()
    if sym not in _CF_BRANCH_COEFFS:
        raise ValueError(f"Unknown symmetry '{sym}'. Use 'oh', 'd4h', or 'c4h'.")

    coeffs = _CF_BRANCH_COEFFS[sym]
    # X40 branch: coefficient × (10Dq - 35dt/6) / 10
    # (write_BAN.py order_cf: X = 10Dq - 35dt/6)
    tendq_effective = tendq - 35.0 * dt / 6.0 if sym != "oh" else tendq
    result = {}
    if "X40" in coeffs:
        result["X40"] = coeffs["X40"] * tendq_effective / 10.0
    if "X420" in coeffs:
        result["X420"] = coeffs["X420"] * dt
    if "X220" in coeffs:
        result["X220"] = coeffs["X220"] * ds
    return result


