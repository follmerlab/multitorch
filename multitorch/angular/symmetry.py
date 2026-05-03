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


# ─────────────────────────────────────────────────────────────
# D4h Butler-label conventions
# ─────────────────────────────────────────────────────────────
#
# The D4h irreps in Butler's notation, as used by ttrac and visible in
# the bundled `nid8.rme_rac` fixture. Each irrep has a Butler label
# (without parity suffix) plus a parity sign appended.
#
# Mapping derived from inspecting nid8.rme_rac IRREP lines and
# cross-referencing with the OH_TO_D4H subduction:
#
#   A1g (Oh '0+')   → D4h '0+'
#   A2g (Oh '^0+')  → D4h '^0+'   (starred / "hat" in Butler)
#   Eg  (Oh '1+')   → D4h '1+'    (dim=2)
#   B1g (Oh '2+')   → D4h '2+'
#   B2g (Oh '^2+')  → D4h '^2+'
# Same pattern with '-' suffix for ungerade.

D4H_TO_BUTLER: Dict[str, str] = {
    'A1g': '0+',  'A1u': '0-',
    'A2g': '^0+', 'A2u': '^0-',
    'Eg':  '1+',  'Eu':  '1-',
    'B1g': '2+',  'B1u': '2-',
    'B2g': '^2+', 'B2u': '^2-',
}

D4H_IRREP_DIM: Dict[str, int] = {
    'A1g': 1, 'A1u': 1,
    'A2g': 1, 'A2u': 1,
    'Eg':  2, 'Eu':  2,
    'B1g': 1, 'B1u': 1,
    'B2g': 1, 'B2u': 1,
}


def d4h_butler_label(d4h_irrep: str) -> str:
    """Return the Butler-style label for a D4h irrep ('A1g' → '0+', etc.).

    Used by the rac_generator dispatcher when emitting GROUND/EXCITE
    blocks under sym='d4h' to match the labeling convention of the
    bundled nid8 fixture.
    """
    if d4h_irrep not in D4H_TO_BUTLER:
        raise ValueError(
            f"Unknown D4h irrep {d4h_irrep!r}; expected one of "
            f"{sorted(D4H_TO_BUTLER)}"
        )
    return D4H_TO_BUTLER[d4h_irrep]


# ─────────────────────────────────────────────────────────────
# D4 (proper-rotation subgroup of D4h) character table
# ─────────────────────────────────────────────────────────────
#
# Conjugacy classes (in order): E, C2_z, 2C4_z, 2C2'_axis, 2C2''_diag
# Irreps: A1, A2, B1, B2, E (all also exist as A1g/A1u etc. in D4h
# via the parity index; for proper rotations we just use D4).

_D4_CONJ_CLASS_ORDER = ['E', 'C2_z', 'C4_z', 'C2_axis', 'C2_diag']

D4_CHARACTERS: Dict[str, Dict[str, float]] = {
    'A1': {'E': 1, 'C2_z': 1,  'C4_z': 1,  'C2_axis': 1,  'C2_diag': 1},
    'A2': {'E': 1, 'C2_z': 1,  'C4_z': 1,  'C2_axis': -1, 'C2_diag': -1},
    'B1': {'E': 1, 'C2_z': 1,  'C4_z': -1, 'C2_axis': 1,  'C2_diag': -1},
    'B2': {'E': 1, 'C2_z': 1,  'C4_z': -1, 'C2_axis': -1, 'C2_diag': 1},
    'E':  {'E': 2, 'C2_z': -2, 'C4_z': 0,  'C2_axis': 0,  'C2_diag': 0},
}


def _classify_d4_rotation(R) -> str:
    """Classify a rotation R (3x3) into one of D4's 5 conjugacy classes.

    Assumes R has been pre-filtered to be in the D4_z subgroup (preserves
    z-axis up to sign). Returns one of: 'E', 'C2_z', 'C4_z', 'C2_axis',
    'C2_diag'.
    """
    import numpy as _np
    tr = _np.trace(R)
    if abs(tr - 3.0) < 0.1:
        return 'E'
    if abs(tr - 1.0) < 0.1:
        return 'C4_z'   # rotation by ±π/2 about z
    if abs(tr + 1.0) < 0.1:
        # C2 — distinguish C2_z (z fixed) from C2_axis (x or y fixed)
        # from C2_diag (diagonal in xy fixed)
        diag = _np.diag(R)
        if abs(diag[2] - 1.0) < 0.1:
            return 'C2_z'    # z-component is +1 → z-axis is rotation axis
        if abs(diag[0] - 1.0) < 0.1 or abs(diag[1] - 1.0) < 0.1:
            return 'C2_axis'  # rotation about x or y
        return 'C2_diag'      # diagonal axis in xy plane
    raise ValueError(f"Unexpected D4 trace: {tr}")


def _is_d4z_rotation(R) -> bool:
    """Check if a 3x3 rotation R preserves the z-axis up to sign.

    R is in the D4_z subgroup of Oh iff R @ (0,0,1) = ±(0,0,1).
    """
    import numpy as _np
    z_image = R @ _np.array([0.0, 0.0, 1.0])
    return abs(abs(z_image[2]) - 1.0) < 0.05 and abs(z_image[0]) < 0.05 and abs(z_image[1]) < 0.05


def oh_to_d4h_subduction_matrix(oh_irrep: str) -> Dict[str, "np.ndarray"]:
    """Return D4h subduction of an Oh single-group irrep.

    For each Oh irrep, computes the unitary rotation that takes the
    Oh-irrep partner basis into D4h-irrep partner basis. Result is
    keyed by D4h irrep (with parity matching the Oh input — Oh g
    irrep maps to D4h g irreps, Oh u maps to D4h u).

    Algorithm:
      1. Identify the 8 D4_z rotations within Oh's 24 (those that
         preserve the z-axis up to sign)
      2. Restrict the Oh irrep matrices to those 8 rotations
      3. For each D4h irrep that appears in the OH_TO_D4H reduction:
         build the D4 character projector and find the eigenvectors
         with eigenvalue 1 — those are the partner basis vectors

    Parameters
    ----------
    oh_irrep : str
        Oh irrep label with parity, e.g. 'A1g', 'Eg', 'T2u'.

    Returns
    -------
    Dict[d4h_irrep, partner_matrix]
        partner_matrix has shape (dim_Oh, mult × dim_D4h) where
        each column is a partner basis vector in the original
        Oh-irrep's basis.

    For Oh single-group irreps only (A1, A2, E, T1, T2 with g/u).
    Half-integer (double-group) irreps are not handled here; they
    require D4h double-group tables.
    """
    import numpy as _np
    from multitorch.angular.point_group import (
        OH_IRREP_DIM,
        _oh_irrep_matrices_real_std,
        octahedral_rotations,
    )

    # Strip parity suffix to look up Oh single-group matrices
    if oh_irrep.endswith('g') or oh_irrep.endswith('u'):
        oh_label = oh_irrep[:-1]
        parity = oh_irrep[-1]
    else:
        raise ValueError(f"oh_irrep must include g/u suffix; got {oh_irrep!r}")

    if oh_label not in OH_IRREP_DIM:
        raise ValueError(
            f"Unsupported Oh irrep {oh_irrep!r}; expected A1g/A2g/Eg/T1g/T2g "
            f"(or _u variants)"
        )

    # 1. Identify the 8 D4_z rotations
    rotations = octahedral_rotations()
    d4_indices = [i for i, R in enumerate(rotations) if _is_d4z_rotation(R)]
    if len(d4_indices) != 8:
        raise RuntimeError(
            f"Expected 8 D4 rotations within Oh's 24, got {len(d4_indices)}"
        )

    # 2. Restrict Oh irrep matrices to D4 rotations
    oh_mats = _oh_irrep_matrices_real_std()[oh_label]
    d4_mats = [oh_mats[i] for i in d4_indices]
    d4_classes = [_classify_d4_rotation(rotations[i]) for i in d4_indices]

    # 3. Build projectors and find partner vectors per D4h irrep
    target_d4h_irreps = OH_TO_D4H.get(oh_irrep, [])
    if not target_d4h_irreps:
        raise ValueError(f"OH_TO_D4H has no entry for {oh_irrep}")

    result: Dict[str, _np.ndarray] = {}
    dim_oh = OH_IRREP_DIM[oh_label]
    for d4h_irrep in target_d4h_irreps:
        d4_irrep = d4h_irrep[:-1]  # strip parity (g/u match Oh's)
        if d4_irrep not in D4_CHARACTERS:
            raise ValueError(f"D4 irrep {d4_irrep!r} not in character table")
        chars = D4_CHARACTERS[d4_irrep]
        dim_d4 = D4H_IRREP_DIM[d4h_irrep]

        # P^Γ = (dim_Γ / |G|) Σ χ^Γ(R)* Γ(R)
        # For D4 |G|=8, characters are real, so χ* = χ
        P = _np.zeros((dim_oh, dim_oh), dtype=_np.float64)
        for i in range(8):
            P += chars[d4_classes[i]] * d4_mats[i]
        P *= dim_d4 / 8.0

        # Eigenvectors with eigenvalue ≈ 1 are the partner vectors
        eigvals, eigvecs = _np.linalg.eigh(P)
        mask = eigvals > 0.5
        partners = eigvecs[:, mask]  # (dim_oh, mult * dim_d4)

        # Verify dimension matches expected mult × dim_d4
        expected_n = OH_TO_D4H[oh_irrep].count(d4h_irrep) * dim_d4
        if partners.shape[1] != expected_n:
            raise RuntimeError(
                f"Partner extraction failed for {oh_irrep} → {d4h_irrep}: "
                f"got {partners.shape[1]} partners, expected {expected_n}"
            )
        result[d4h_irrep] = partners

    return result


def d4h_partner_basis_per_J(J, d4h_irrep: str) -> "np.ndarray":
    """Return partner-basis matrix (2J+1, n_partners) for D^J → D4h irrep.

    Builds the basis by composing two subductions:

      1. ``_real_subduction_matrix(J, oh_irrep)`` projects D^J onto each
         contributing Oh irrep.
      2. ``oh_to_d4h_subduction_matrix(oh_irrep)[d4h_irrep]`` rotates
         each Oh-irrep partner basis into D4h-irrep partners.

    The returned matrix has columns ordered by:
    (oh_irrep contributing this d4h_irrep) → (multiplicity copy of oh
    irrep in D^J) → (D4h partner index).

    For example, J=2 → D4h A1g:
    - Oh Eg subduces (1× in J=2) → A1g (1× in Eg) = 1 partner from Eg-A1g path
    - Oh T2g subduces (1× in J=2) → no A1g (T2g → B2g + Eg only)
    Total: 1 A1g partner from J=2.

    Parameters
    ----------
    J : int
        Angular momentum (integer; half-integer raises NotImplementedError
        per d4h_irreps_for_J).
    d4h_irrep : str
        Target D4h irrep ('A1g', 'A2g', 'B1g', 'B2g', 'Eg' and 'u' variants).

    Returns
    -------
    np.ndarray of shape (2J+1, n_partners)
        Each column is an orthonormal basis vector in the real spherical
        harmonic basis of D^J. n_partners = mult × dim(d4h_irrep) where
        mult is the number of times d4h_irrep appears in the
        D^J → D4h reduction.
    """
    import numpy as _np
    from multitorch.angular.point_group import (
        OH_IRREP_DIM, _real_subduction_matrix, oh_branching,
    )

    is_half_int = abs(J - round(J)) > 0.1
    if is_half_int:
        raise NotImplementedError(
            "d4h_partner_basis_per_J for half-integer J needs D4h "
            "double-group tables (not yet tabulated)."
        )

    J_int = int(round(J))
    parity_suffix = 'g' if (J_int % 2 == 0) else 'u'

    # Strip parity to compare against d4h_irrep's parity
    d4h_parity = d4h_irrep[-1]
    if d4h_parity != parity_suffix:
        # Wrong-parity D4h irrep can't appear in D^J
        return _np.zeros((2 * J_int + 1, 0), dtype=_np.float64)

    # For each Oh irrep that subduces to this d4h_irrep, collect partners
    cols: List[_np.ndarray] = []
    for oh_label, dim_oh in OH_IRREP_DIM.items():
        oh_irrep_full = f'{oh_label}{parity_suffix}'
        d4h_targets = OH_TO_D4H.get(oh_irrep_full, [])
        if d4h_irrep not in d4h_targets:
            continue
        mult_oh = oh_branching(J_int).get(oh_label, 0)
        if mult_oh == 0:
            continue

        # B_oh has shape (2J+1, dim_oh * mult_oh) with columns ordered by
        # (copy_index, partner). For our purposes we want the partners
        # of each copy, then rotate via oh_to_d4h_subduction_matrix.
        B_oh = _real_subduction_matrix(J_int, oh_label)
        # B_oh columns are grouped per copy: copy_0_partner_0, copy_0_partner_1,
        # ..., copy_1_partner_0, ...
        sub = oh_to_d4h_subduction_matrix(oh_irrep_full)[d4h_irrep]
        # sub has shape (dim_oh, n_d4h_partners_in_this_oh_irrep)

        for copy_i in range(mult_oh):
            B_oh_copy = B_oh[:, copy_i * dim_oh:(copy_i + 1) * dim_oh]
            # Rotate this copy's partners into D4h basis
            # Result shape: (2J+1, n_d4h_partners_in_this_oh_irrep)
            rotated = B_oh_copy @ sub
            cols.append(rotated)

    if not cols:
        return _np.zeros((2 * J_int + 1, 0), dtype=_np.float64)

    return _np.column_stack(cols)


def d4h_irreps_for_J(J) -> List[Tuple[str, int]]:
    """Return [(D4h_irrep, multiplicity), ...] for D^J in D4h.

    Composes oh_branching(J) (multitorch.angular.point_group) with
    OH_TO_D4H. For example, J=2 in Oh gives Eg + T2g; in D4h that's
    A1g (from Eg) + B1g (from Eg) + B2g (from T2g) + Eg (from T2g).

    Parity is determined by J's natural parity ((-1)^L for orbital).
    """
    from multitorch.angular.point_group import oh_branching

    is_half_int = abs(J - round(J)) > 0.1
    if is_half_int:
        raise NotImplementedError(
            "d4h_irreps_for_J for half-integer J requires D4h "
            "double-group tables (not yet tabulated)."
        )

    oh_b = oh_branching(int(round(J)))
    parity_suffix = 'g' if (int(round(J)) % 2 == 0) else 'u'

    result: Dict[str, int] = {}
    for oh_irrep, mult in oh_b.items():
        if mult == 0:
            continue
        oh_irrep_full = f'{oh_irrep}{parity_suffix}'
        d4h_irreps = OH_TO_D4H.get(oh_irrep_full, [oh_irrep_full])
        for d4h_irrep in d4h_irreps:
            result[d4h_irrep] = result.get(d4h_irrep, 0) + mult
    # Return as sorted list for determinism
    return sorted(result.items(), key=lambda kv: list(D4H_TO_BUTLER).index(kv[0]))


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
