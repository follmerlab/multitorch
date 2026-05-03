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
from typing import Dict, List, Tuple  # noqa: F401  (re-exported in type hints below)
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
# Butler branch coefficients for D4h crystal-field operators
# ─────────────────────────────────────────────────────────────
#
# The D4h L-edge ttrac path uses three crystal-field scalar operators
# beyond the trivial HAMILTONIAN (k=0):
#
#   TENDQ — rank-4, projects to A1g of D4h via the Oh A1g branch
#   DT    — rank-4, projects to A1g + Eg of D4h
#   DS    — rank-2, projects to A1g of D4h
#
# Per pyctm/write_RAC.py:gen_X400_X420_X220 (lines 349-351), the
# COWAN-store coefficients written by Fortran are:
#
#   X400 = (6/10) * sqrt(30) * tendq  −  (7/2) * sqrt(30) * dt
#   X420 = -(5/2) * sqrt(42) * dt
#   X220 = -sqrt(70) * ds
#
# X400 is the (4+,0+,0+) branch coefficient; X420 is (4+,2+,2+);
# X220 is (2+,0+,0+). Each operator contributes a (possibly multi-
# branch) set of unit-scaled coefficients; the atomic-parameter scalars
# (tendq, dt, ds) come in via XHAM at BAN-construction time, so the
# tables here encode unit values only.
#
# Schema: D4H_BRANCHES_BY_OPERATOR[op_name][(O3_irrep, Oh_irrep, D4h_irrep)]
# = unit branch coefficient (i.e., the value when the corresponding
# atomic parameter equals 1.0).

# Butler keys are (O3_irrep, Oh_irrep_path, D4h_irrep_target) where each
# label is in J-numbered Butler form. The third slot is always '0+' for
# CF operators because they're scalar (D4h A1g) — they only differ in
# rank and Oh-route. Verbatim mapping from the BRANCH lines in
# pyctm/write_RAC.py:shellDict['d4h']:
#
#   'BRANCH 4+ > 0 0+ > 0+ <coeff>'   (X400 tendq part)
#   'BRANCH 4+ > 0 2+ > 0+ <coeff>'   (X420 dt part — O3-rank-4 via Oh-2+/E)
#   'BRANCH 2+ > 0 2+ > 0+ <coeff>'   (X220 ds — O3-rank-2 via Oh-2+/E)
#
# X400 = (6/10)*sqrt(30)*tendq − (7/2)*sqrt(30)*dt → both share the
# ('4+','0+','0+') key, hence the per-operator schema.

D4H_BRANCHES_BY_OPERATOR: Dict[str, Dict[Tuple[str, str, str], float]] = {
    'TENDQ': {
        # X400 contribution from tendq, via Oh A1 → D4h A1g
        ('4+', '0+', '0+'): 6.0 * math.sqrt(30.0) / 10.0,
    },
    'DT': {
        # X400 contribution from dt, via Oh A1 → D4h A1g
        ('4+', '0+', '0+'): -7.0 / 2.0 * math.sqrt(30.0),
        # X420 contribution from dt, via Oh E → D4h A1g
        ('4+', '2+', '0+'): -5.0 / 2.0 * math.sqrt(42.0),
    },
    'DS': {
        # X220 contribution from ds, via Oh E → D4h A1g
        ('2+', '2+', '0+'): -math.sqrt(70.0),
    },
}


# DEPRECATED: legacy single-coefficient table that conflated TENDQ and DT
# into one entry per (O3, Oh, D4h) triplet. Two of three values were also
# numerically wrong vs pyctm. Kept as a compatibility re-export populated
# from D4H_BRANCHES_BY_OPERATOR['TENDQ'] (the only operator with a single
# branch); new code should use D4H_BRANCHES_BY_OPERATOR.
D4H_SHELL_BRANCHES: Dict[Tuple[str, str, str], float] = dict(
    D4H_BRANCHES_BY_OPERATOR['TENDQ']
)


def d4h_cf_operator_recipe(operator: str) -> List[Tuple[int, str, str, float]]:
    """Return the COWAN-block recipe for one D4h crystal-field operator.

    Each entry in the returned list says: "to assemble this D4h operator's
    contribution, use the rank-K SHELL coupling for Oh single-group irrep
    `oh_irrep`, route the result to D4h irrep `d4h_irrep`, and scale by
    `branch_coeff` (the value when the atomic parameter equals 1.0)."

    The atomic parameter scalar (tendq, dt, or ds) is applied separately
    via the BAN XHAM mechanism, so the branch coefficients here are
    parameter-independent.

    Parameters
    ----------
    operator : {'TENDQ', 'DT', 'DS'}
        The D4h CF operator name.

    Returns
    -------
    list of (rank, oh_irrep, d4h_irrep, branch_coeff) tuples
        rank: O3 rank (4 for tendq/dt, 2 for ds)
        oh_irrep: Oh single-group irrep label without parity ('A1' or 'E')
        d4h_irrep: D4h irrep label with parity ('A1g', 'Eg', etc.)
        branch_coeff: unit-parameter Butler branch coefficient
    """
    if operator not in D4H_BRANCHES_BY_OPERATOR:
        raise ValueError(
            f"Unknown D4h CF operator {operator!r}; "
            f"available: {sorted(D4H_BRANCHES_BY_OPERATOR)}"
        )

    # O3-irrep label → operator rank (the L value of the rank-K spherical
    # tensor). Butler labels carry parity, but for CF (always gerade) the
    # parity is +.
    o3_to_rank = {'0+': 0, '2+': 2, '4+': 4}

    # The middle slot of the Butler key (Oh_irrep) is in J-numbered Butler
    # form ('0+' = A1, '2+' = E in the relevant decomposition). Map to the
    # standard Oh single-group label that oh_coupling_coefficients returns.
    # NOTE: For the CF operators in this table only A1 and E paths appear;
    # the trickier '4+' slot mapping is unused here.
    butler_oh_to_label = {'0+': 'A1', '2+': 'E'}

    # All CF operators are scalar (A1g of D4h) — they're invariant under
    # D4h symmetry operations. The BRANCH lines in pyctm/write_RAC.py all
    # target '0+' on the D4h side (= D4h A1g). What differs between
    # operators is the rank K and the Oh-irrep route.
    recipe: List[Tuple[int, str, str, float]] = []
    branches = D4H_BRANCHES_BY_OPERATOR[operator]
    for (o3_irr, oh_irr_butler, d4h_irr_butler), coeff in branches.items():
        rank = o3_to_rank[o3_irr]
        oh_irr = butler_oh_to_label[oh_irr_butler]
        assert d4h_irr_butler == '0+', (
            f"Unexpected D4h target {d4h_irr_butler!r} in {operator} table; "
            "all CF operators should target D4h A1g (Butler '0+')."
        )
        recipe.append((rank, oh_irr, 'A1g', coeff))
    return recipe


def d4h_branching(J) -> Dict[str, int]:
    """Compute the O3 → D4h branching for angular momentum J.

    Returns the multiplicity of each D4h irrep in the reduction of D^J,
    obtained by composing oh_branching(J) with the OH_TO_D4H subduction
    table. Both even-parity (gerade) and odd-parity (ungerade) branches
    are handled — the parity is carried by the Oh irrep label, which
    OH_TO_D4H maps to the corresponding parity-suffixed D4h irrep.

    Parameters
    ----------
    J : int or float
        Non-negative angular momentum. Integer J branches into
        single-group D4h irreps (A1g, A2g, B1g, B2g, Eg and their
        ungerade counterparts). Half-integer J support requires
        D4h double-group branching, which is not yet tabulated here.

    Returns
    -------
    dict mapping D4h irrep name → multiplicity
    """
    from multitorch.angular.point_group import oh_branching

    is_half_int = abs(J - round(J)) > 0.1
    if is_half_int:
        raise NotImplementedError(
            "d4h_branching for half-integer J requires D4h double-group "
            "tables which are not yet implemented. Track in plan Phase 1c."
        )

    oh_b = oh_branching(int(round(J)))

    # OH_TO_D4H is keyed by Oh single-group irreps; for J branched into
    # ungerade Oh irreps (T1u, T2u, etc.) the mapping is symmetric.
    result: Dict[str, int] = {}
    for oh_irrep, mult in oh_b.items():
        if mult == 0:
            continue
        # oh_branching returns single-group labels without g/u suffix
        # (e.g., 'A1', 'T1'); apply parity from J's natural parity.
        # For integer L, parity is (-1)^L: even L → g, odd L → u.
        parity_suffix = 'g' if (int(round(J)) % 2 == 0) else 'u'
        oh_irrep_full = f'{oh_irrep}{parity_suffix}'
        d4h_irreps = OH_TO_D4H.get(oh_irrep_full, [oh_irrep_full])
        for d4h_irrep in d4h_irreps:
            result[d4h_irrep] = result.get(d4h_irrep, 0) + mult
    return result


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
