"""
Group theory: O3 → Oh → D4h → C4h subduction coefficients.

These are the branching rules for how irreducible representations of
larger groups decompose into irreps of smaller subgroups. They are
needed to:
  1. Convert ttrcg Oh-symmetry RME (.m14) to D4h RME (.m15) [ttrac]
  2. Apply crystal field splitting in lower symmetries

The branching coefficients are exact algebraic values from:
  Butler (1981) Point Group Symmetry Applications.
  The tables are encoded in the rcg_cfp files (groups loaded by ttrac).

Current status: tabulated for the most common d-electron symmetry chain
O3 → Oh → D4h, based on values in write_RAC.py.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import math


# ─────────────────────────────────────────────────────────────
# Irrep labels for each group
# ─────────────────────────────────────────────────────────────

# O3 irreps relevant to d-electrons: L=0,1,2,3,4 × parity
O3_IRREPS_D = ['0+', '1-', '2+', '3-', '4+', '1+', '2-', '3+', '4-']

# Oh double group irreps (for d-electrons in octahedral field)
OH_IRREPS = {
    '0+': 'A1g',    # L=0+ → A1g
    '2+': 'Eg',     # L=2+ → Eg + T2g
    '4+': 'A1g',    # L=4+ → A1g + Eg + T1g + T2g
    '1-': 'T1u',    # L=1- → T1u
    '3-': 'A2u',    # L=3- → multiple
}

# D4h irreps (for d-electrons in tetragonal field)
# Branching: Oh → D4h
OH_TO_D4H = {
    'A1g': ['A1g'],
    'A2g': ['B1g'],
    'Eg': ['A1g', 'B1g'],
    'T1g': ['A2g', 'Eg'],
    'T2g': ['B2g', 'Eg'],
    'A1u': ['A1u'],
    'A2u': ['B1u'],
    'Eu': ['A1u', 'B1u'],
    'T1u': ['A2u', 'Eu'],
    'T2u': ['B2u', 'Eu'],
}


# ─────────────────────────────────────────────────────────────
# Butler branch coefficients for D4h
# These are the same values as in write_RAC.py shellDict/hamDict
# ─────────────────────────────────────────────────────────────

# Branch coefficients for the crystal field SHELL operators in D4h:
# Each entry: (O3_irrep, Oh_irrep, D4h_irrep): coefficient
D4H_SHELL_BRANCHES = {
    # SHELL2 (quadrupole, rank-2): ds parameter
    ('2+', '0+', '0+'): -math.sqrt(70.0) / 10.0,    # B2g branch coefficient
    # SHELL1 (quadrupole, rank-4): dt and 10Dq parameters
    ('4+', '0+', '0+'): 3.0 * math.sqrt(30.0) / 5.0,    # A1g branch
    ('4+', '2+', '2+'): -5.0 * math.sqrt(42.0) / (2.0 * 7.0),   # Eg branch
}

# These match the exact BRANCH entries in write_RAC.py:
# BRANCH 4+ > 0 0+ > 0+   3.28633   (= 3*sqrt(30)/5 = 6*sqrt(30)/10)
# BRANCH 4+ > 0 2+ > 0+   0.0        (for Dt only: -16.2019)
# BRANCH 2+ > 0 2+ > 0+   -0.8367   (= -sqrt(70)/10)


def get_oh_irreps_from_o3(j: int, parity: str) -> List[str]:
    """
    Get Oh irreps from O3 irrep (j, parity).

    Parameters
    ----------
    j : int
        Angular momentum L (0=S, 1=P, 2=D, 3=F, 4=G).
    parity : str
        '+' or '-'.

    Returns
    -------
    List of Oh irrep labels.
    """
    # Branching rules O3 → Oh (well-known from group theory)
    # Even parity (g = gerade)
    oh_even = {
        0: ['A1g'],
        1: ['T1g'],
        2: ['Eg', 'T2g'],
        3: ['A2g', 'T1g', 'T2g'],
        4: ['A1g', 'Eg', 'T1g', 'T2g'],
    }
    # Odd parity (u = ungerade)
    oh_odd = {
        0: ['A1u'],
        1: ['T1u'],
        2: ['Eu', 'T2u'],
        3: ['A2u', 'T1u', 'T2u'],
        4: ['A1u', 'Eu', 'T1u', 'T2u'],
    }
    if parity == '+':
        return oh_even.get(j, [])
    else:
        return oh_odd.get(j, [])
