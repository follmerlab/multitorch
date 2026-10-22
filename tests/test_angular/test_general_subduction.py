"""The general Oh → G subduction, and its gate against the tetragonal one.

``oh_to_d4h_subduction_matrix`` was written first and is Fortran-validated
through the D4h ADD coefficients it feeds. ``oh_to_group_subduction_matrix`` is
the same construction with the D4-specific pieces -- the rotation filter, the
conjugacy classes, the characters and the branching table -- taken from
``point_group(H)`` instead, and with a dihedral group's 2-D irrep realised
geometrically as the action on the plane perpendicular to the principal axis.

Two kinds of test here, and the distinction matters:

* The **gate**: the general function must reproduce the tetragonal one *bit for
  bit*, which is what makes routing D4h through it a relabelling rather than a
  recomputation and so leaves D4h's Fortran parity untouched. Exact equality is
  a licence, not a physics oracle (CLAUDE.md), so it is paired with --
* the **independent checks**: that the partner bases actually block-diagonalise
  the group, and that the Oh → D3d branching composes with ``oh_branching`` to
  give the direct D^J → D3d subduction. Those two are statements about the
  group, not about either implementation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.finite_group import PRINCIPAL_AXIS, point_group
from multitorch.angular.point_group import (
    OH_IRREP_DIM,
    _oh_irrep_matrices_real_std,
    oh_branching,
    octahedral_rotations,
)
from multitorch.angular import symmetry
from multitorch.angular.symmetry import (
    OH_ROTATION_SUBGROUP,
    OH_TO_D4H,
    SUBDUCTION_TARGETS,
    oh_to_d4h_subduction_matrix,
    oh_to_group_subduction_matrix,
    oh_to_target_subduction_matrix,
    target_irrep_dim,
)

OH_SINGLE_IRREPS = tuple(sorted(OH_TO_D4H))          # all ten, g and u


# ── the gate ─────────────────────────────────────────────────

@pytest.mark.parametrize("oh_irrep", OH_SINGLE_IRREPS)
def test_the_general_subduction_reproduces_the_tetragonal_one_exactly(oh_irrep):
    """Bit for bit, keys included -- so D4h's ADD coefficients cannot move."""
    reference = oh_to_d4h_subduction_matrix(oh_irrep)
    general = oh_to_group_subduction_matrix(oh_irrep, 'D4h')

    assert set(general) == set(reference), (
        f"{oh_irrep}: general gives {sorted(general)}, "
        f"tetragonal gives {sorted(reference)}")
    for key in reference:
        assert general[key].shape == reference[key].shape, key
        assert np.array_equal(general[key], reference[key]), (
            f"{oh_irrep} → {key}: max difference "
            f"{np.abs(general[key] - reference[key]).max():.3e}, expected 0.0")


def test_without_the_character_denoising_the_equality_is_only_approximate():
    """The injection that shows the gate is reading the last bit.

    ``point_group``'s character table is computed, so a character that is
    exactly ±1 arrives as 1 - 2.2e-16. With that noise left in, the general
    construction still agrees with the tetragonal one to 3.3e-16 -- which is
    *correct* but not a relabelling, and would have moved every published D4h
    coefficient in its last decimal. This records which of the two the gate is
    measuring.
    """
    original = symmetry._denoise_character
    symmetry._denoise_character = lambda x, tol=1e-9: float(x)
    try:
        worst = 0.0
        for oh_irrep in OH_SINGLE_IRREPS:
            reference = oh_to_d4h_subduction_matrix(oh_irrep)
            general = oh_to_group_subduction_matrix(oh_irrep, 'D4h')
            for key in reference:
                worst = max(worst, float(np.abs(general[key] - reference[key]).max()))
    finally:
        symmetry._denoise_character = original

    assert worst > 0.0, "denoising made no difference; the injection is inert"
    assert worst < 1e-14, f"difference {worst:.3e} is larger than rounding noise"

    # and with it back, exact again
    for oh_irrep in OH_SINGLE_IRREPS:
        reference = oh_to_d4h_subduction_matrix(oh_irrep)
        general = oh_to_group_subduction_matrix(oh_irrep, 'D4h')
        assert all(np.array_equal(general[k], reference[k]) for k in reference)


def test_swapping_the_perpendicular_frame_breaks_the_equality():
    """The second injection: the 2-D irrep's gauge is load-bearing.

    The geometric realisation of a dihedral E is only the tetragonal code's
    ``R[:2, :2]`` because the frame is (x, y) in that order. Swap the two
    columns and the partner basis is a different one -- still a valid
    subduction, which is the point: the gate is pinning *which* one.
    """
    original = symmetry._perpendicular_frame
    symmetry._perpendicular_frame = lambda axis: original(axis)[:, ::-1]
    try:
        differs = []
        for oh_irrep in OH_SINGLE_IRREPS:
            reference = oh_to_d4h_subduction_matrix(oh_irrep)
            general = oh_to_group_subduction_matrix(oh_irrep, 'D4h')
            if any(not np.array_equal(general[k], reference[k]) for k in reference):
                differs.append(oh_irrep)
    finally:
        symmetry._perpendicular_frame = original

    # the four Oh irreps that carry a D4h E are the ones that can notice
    assert sorted(differs) == ['T1g', 'T1u', 'T2g', 'T2u'], differs


def test_the_dispatcher_now_calls_the_general_construction():
    """Records what the function was handed, not that the source says so.

    Trap 3: a source-text assertion is not evidence the line runs.
    """
    calls = []
    original = symmetry.oh_to_group_subduction_matrix

    def recording(oh_irrep, target_group):
        calls.append((oh_irrep, target_group))
        return original(oh_irrep, target_group)

    symmetry.oh_to_group_subduction_matrix = recording
    try:
        oh_to_target_subduction_matrix('T2g', 'D4h')
        oh_to_target_subduction_matrix('T2g', 'Oh')       # identity branch
    finally:
        symmetry.oh_to_group_subduction_matrix = original

    assert calls == [('T2g', 'D4h')], calls


# ── independent checks on the group, not on either implementation ──

@pytest.mark.parametrize("target_group", ['D4h', 'D3d'])
@pytest.mark.parametrize("oh_irrep", OH_SINGLE_IRREPS)
def test_the_partner_basis_block_diagonalises_the_target_group(target_group,
                                                               oh_irrep):
    """S† D(R) S is the direct sum of Γ(R) over the copies, for every rotation.

    This is what "subduction" means, stated without reference to either
    function's internals: the columns are an orthonormal basis of the Oh irrep
    in which every element of the target group acts as copies of one target
    irrep. It is also the check that the geometric 2-D construction picked the
    right representation.
    """
    group = point_group(OH_ROTATION_SUBGROUP[target_group])
    pairs = symmetry._target_rotations_in_oh(group)
    oh_mats = _oh_irrep_matrices_real_std()[oh_irrep[:-1]]

    for label, S in oh_to_group_subduction_matrix(oh_irrep, target_group).items():
        stem = label[:-1]
        gamma = symmetry._target_irrep_matrices(group, stem, pairs)
        dim_t = gamma[0].shape[0]
        mult = S.shape[1] // dim_t

        assert np.allclose(S.T @ S, np.eye(S.shape[1]), atol=1e-10), (
            f"{oh_irrep} → {label}: partner basis is not orthonormal")

        for k, (oh_index, _gi) in enumerate(pairs):
            block = S.T @ oh_mats[oh_index] @ S
            expected = np.kron(np.eye(mult), gamma[k])
            assert np.allclose(block, expected, atol=1e-9), (
                f"{oh_irrep} → {label}, rotation {oh_index}: the group does not "
                f"act as {mult} copies of {stem}")


@pytest.mark.parametrize("J", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("parity", [+1, -1])
def test_the_oh_to_d3d_branching_composes_to_the_direct_subduction(J, parity):
    """D^J → D3d, two ways: directly, and through Oh.

    The direct route is ``point_group('D3d').branching``, which never mentions
    Oh. The composed route is ``oh_branching(J)`` -- the Oh content of D^J,
    which is parity-blind, so the parity only decides the g/u suffix --
    followed by the Oh → D3d subduction under test. They have to agree, and
    neither is derived from the other.
    """
    d3d = point_group('D3d')
    direct = d3d.branching(J, parity)

    suffix = 'g' if parity > 0 else 'u'
    composed: dict = {}
    for oh_stem, mult in oh_branching(J).items():
        if mult == 0:
            continue
        oh_irrep = oh_stem if oh_stem[-1] in 'gu' else oh_stem + suffix
        sub = oh_to_group_subduction_matrix(oh_irrep, 'D3d')
        for label, S in sub.items():
            copies = S.shape[1] // target_irrep_dim(label, 'D3d')
            composed[label] = composed.get(label, 0) + mult * copies

    assert {k: v for k, v in composed.items() if v} == \
           {k: v for k, v in direct.items() if v}, (
        f"J={J} parity={parity}: through Oh {composed}, directly {direct}")


def test_the_trigonal_descent_is_the_familiar_one():
    """t2g → a1g + eg and eg → eg, which is what a trigonal distortion does.

    Spelled out rather than computed so that a change of convention has to
    walk past a human-readable statement of the physics.
    """
    expected = {
        'A1g': {'A1g': 1},
        'A2g': {'A2g': 1},
        'Eg':  {'Eg': 1},
        'T1g': {'A2g': 1, 'Eg': 1},
        'T2g': {'A1g': 1, 'Eg': 1},
    }
    for oh_irrep, want in expected.items():
        sub = oh_to_group_subduction_matrix(oh_irrep, 'D3d')
        got = {label: S.shape[1] // target_irrep_dim(label, 'D3d')
               for label, S in sub.items()}
        assert got == want, f"{oh_irrep}: {got}, expected {want}"
        for label, S in sub.items():
            assert S.shape[0] == OH_IRREP_DIM[oh_irrep[:-1]], (oh_irrep, label)


# ── refusals ─────────────────────────────────────────────────

def test_a_trigonal_group_in_the_z_setting_is_refused():
    """Residual 40's trap, caught at the door.

    A trigonal group in the crystallographic z setting is not a subgroup of the
    standard Oh -- 8 of D3d's 12 elements lie outside it -- so "10Dq plus a
    trigonal distortion" is not expressible there at all. The registry keeps
    both settings on purpose, and this is the error that says which one is in
    play rather than quietly returning a wrong basis.
    """
    OH_ROTATION_SUBGROUP['D3d[z]'] = 'D3[z]'
    try:
        with pytest.raises(RuntimeError, match="cubic setting"):
            oh_to_group_subduction_matrix('T2g', 'D3d[z]')
    finally:
        del OH_ROTATION_SUBGROUP['D3d[z]']


def test_an_unregistered_target_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no rotation subgroup"):
        oh_to_group_subduction_matrix('T2g', 'C4v')
    # Oh and Td never reach here: their subduction is the identity
    with pytest.raises(ValueError, match="no rotation subgroup"):
        oh_to_group_subduction_matrix('T2g', 'Oh')


def test_the_oh_irrep_must_carry_its_parity():
    with pytest.raises(ValueError, match="g/u suffix"):
        oh_to_group_subduction_matrix('T2', 'D3d')


def test_d3d_is_a_declared_subduction_target_and_its_dims_are_known():
    assert 'D3d' in SUBDUCTION_TARGETS
    assert [target_irrep_dim(i, 'D3d') for i in ('A1g', 'A2g', 'Eg')] == [1, 1, 2]
    assert [target_irrep_dim(i, 'D3d') for i in ('A1u', 'A2u', 'Eu')] == [1, 1, 2]


# ── Butler's own O ⇒ D3 table, single and double group (residual 60) ──
#
# The subduction above was checked against ``point_group`` and against the
# descent limit. Neither is the table ttrac itself uses. Butler's is, and it is
# in the tree:
#
#   SOURCE   : Butler, *Point Group Symmetry Applications: Methods and Tables*
#              (Plenum, 1981), the O ⇒ D3 branching table.
#   LOCATION : ``ttmult/src/disk0``, block ``$F3/OD3``, the ``BRANCHINGS`` line
#              (line 1654 of the file as committed); header reads
#              "BRANCHDATA O=>D3 CREATED FOR_BUTLER(1981) 28_MAR_1980".
#   METHOD   : transcribed from the ASCII block, not re-derived.
#   TIER     : direct quote from a machine-readable table shipped with the
#              Fortran suite, so the same class as a tabulated value.
#
# Verbatim:
#
#   BRANCHINGS 0 +0 ; S0 +S0 ; 1 +^0 +1 ; S1 +S0 +S1 +-S1 ;
#              2 +1 ; ^1 +0 +1 ; ^S0 +S0 ; ^0 +^0 ; ENDBRANCHINGS
#
# Butler's O labels are the ones ``derive_butler_routes.py`` already uses:
# ``0``=A1, ``^0``=A2, ``2``=E, ``1``=T1, ``^1``=T2, and for the double group
# ``S0``=E1/2 (Γ6), ``^S0``=E5/2 (Γ7), ``S1``=G3/2 (Γ8). D3's own labels come
# from the ``GROUPDATA D3`` block's IRREPS line, ``0 +1 S0 -2 ^0 +1 1 +2
# S1 * -S1 -1``: six irreps, of which ``S1`` and ``-S1`` are the
# complex-conjugate pair of **one-dimensional** spinor irreps -- the structural
# difference from D4h, every one of whose double irreps is 2-D.

BUTLER_O_TO_D3 = {
    '0':   ['0'],
    'S0':  ['S0'],
    '1':   ['^0', '1'],
    'S1':  ['S0', 'S1', '-S1'],
    '2':   ['1'],
    '^1':  ['0', '1'],
    '^S0': ['S0'],
    '^0':  ['^0'],
}

#: Oh irrep (this package's labels) → Butler's O label.
OH_TO_BUTLER_O = {
    'A1': '0', 'A2': '^0', 'E': '2', 'T1': '1', 'T2': '^1',
    'E1/2': 'S0', 'E5/2': '^S0', 'G3/2': 'S1',
}

#: Butler's D3 label → this package's D3 label.
BUTLER_D3_TO_LABEL = {
    '0': 'A1', '^0': 'A2', '1': 'E',
    'S0': 'E1/2', 'S1': '1E3/2', '-S1': '2E3/2',
}

#: Oh double-group irreps, in the order ``_oh_double_irrep_matrices`` gives.
OH_DOUBLE_IRREPS = ('E1/2', 'E5/2', 'G3/2')


def _butler_composed(J):
    """D^J → D3 content, routed through Butler's table."""
    composed: dict = {}
    for oh_irrep, mult in oh_branching(J).items():
        if not mult:
            continue
        for butler_d3 in BUTLER_O_TO_D3[OH_TO_BUTLER_O[oh_irrep]]:
            label = BUTLER_D3_TO_LABEL[butler_d3]
            composed[label] = composed.get(label, 0) + mult
    return {k: v for k, v in composed.items() if v}


@pytest.mark.parametrize("two_j", range(0, 12))
def test_butlers_table_composes_to_the_direct_d3_branching(two_j):
    """D^J → D3 two ways: through Butler's O ⇒ D3, and directly.

    The direct route is ``point_group('D3', spinor=True).branching``, which
    knows nothing of Butler or of Oh. The composed route is this package's
    SO3 → Oh branching followed by Butler's transcribed table. Covers integer
    and half-integer J together, so the double-group half of the trigonal
    descent is pinned against the same source ttrac reads.
    """
    J = two_j / 2
    direct = {k: v for k, v in point_group('D3', spinor=True).branching(J).items() if v}
    assert _butler_composed(J) == direct, f"J={two_j}/2"


def test_a_wrong_butler_entry_is_caught():
    """The injection: the composition has to be able to notice a bad route."""
    original = BUTLER_O_TO_D3['^1']
    BUTLER_O_TO_D3['^1'] = ['^0', '1']          # T2 → A2 + E, i.e. T1's row
    try:
        failures = [two_j for two_j in range(12)
                    if _butler_composed(two_j / 2) !=
                    {k: v for k, v in
                     point_group('D3', spinor=True).branching(two_j / 2).items() if v}]
    finally:
        BUTLER_O_TO_D3['^1'] = original
    assert failures, "swapping T2's branching for T1's changed nothing"
    assert 4 in failures, (
        f"J=2, which is where a d shell's crystal field lives, should notice; "
        f"failures were at 2J = {failures}")


@pytest.mark.parametrize("oh_irrep", ['A1g', 'A2g', 'Eg', 'T1g', 'T2g'])
def test_butlers_single_group_rows_are_what_the_subduction_produces(oh_irrep):
    """Butler's integer rows against ``oh_to_group_subduction_matrix``."""
    expected: dict = {}
    for butler_d3 in BUTLER_O_TO_D3[OH_TO_BUTLER_O[oh_irrep[:-1]]]:
        label = BUTLER_D3_TO_LABEL[butler_d3] + 'g'
        expected[label] = expected.get(label, 0) + 1

    got = {label: S.shape[1] // target_irrep_dim(label, 'D3d')
           for label, S in oh_to_group_subduction_matrix(oh_irrep, 'D3d').items()}
    assert got == expected, f"{oh_irrep}: {got}, Butler says {expected}"


def _commutant_blocks(mats):
    """Isotypic blocks of a set of matrices, via a generic commutant element.

    Test-only, and independent of anything in the package: it decomposes the
    restriction directly rather than asking a subduction function. The
    commutation is **asserted**, because the column-major form of the vec
    identity is the natural thing to write, is wrong for numpy's row-major
    ``reshape``, and still returns a plausible block split -- it produced
    exactly Butler's ``[1, 1, 2]`` for Γ8 ⇓ D3 out of matrices that commuted
    with nothing (residual 60).
    """
    n = mats[0].shape[0]
    rows = []
    for D in mats:
        rows.append(np.kron(np.eye(n), D.T) - np.kron(D, np.eye(n)))
    _u, s, vh = np.linalg.svd(np.vstack(rows))
    null = vh[int(np.sum(s > 1e-9)):].conj().T
    basis = [null[:, k].reshape(n, n) for k in range(null.shape[1])]

    for X in basis:
        for D in mats:
            assert np.allclose(X @ D, D @ X, atol=1e-9), (
                "the commutant solve is wrong; see this function's docstring")

    rng = np.random.default_rng(0)
    X = sum(complex(rng.normal(), rng.normal()) * B for B in basis)
    X = 0.5 * (X + X.conj().T)
    vals, vecs = np.linalg.eigh(X)
    blocks, start = [], 0
    for k in range(1, len(vals) + 1):
        if k == len(vals) or abs(vals[k] - vals[start]) > 1e-7:
            blocks.append(vecs[:, start:k])
            start = k
    return blocks


@pytest.mark.parametrize("target,oh_double,expected_dims", [
    ('D4h', 'E1/2', [2]),
    ('D4h', 'E5/2', [2]),
    ('D4h', 'G3/2', [2, 2]),
    ('D3d', 'E1/2', [2]),
    ('D3d', 'E5/2', [2]),
    ('D3d', 'G3/2', [1, 1, 2]),
])
def test_the_double_group_restrictions_split_the_way_butler_says(target, oh_double,
                                                                expected_dims):
    """Γ ⇓ H decomposes into the irreps Butler's table lists, with their dims.

    This is the fact step 2 of the trigonal chain rests on, measured before any
    double-group subduction is written for D3d: ``S1 +S0 +S1 +-S1`` means Oh's
    Γ8 restricted to D3 carries one 2-D and two 1-D irreps, and the two 1-D
    ones occur **nowhere else** -- not in Γ6's restriction and not in Γ7's,
    both of which are irreducible and both of which are E1/2. D4h has no such
    case: all of its double irreps are 2-D and Γ6 ⇓ and Γ7 ⇓ are whole irreps,
    which is why the realization the tetragonal code uses -- take the
    restriction entire -- cannot be carried over unchanged.
    """
    from multitorch.angular.point_group import _oh_double_irrep_matrices

    group = point_group(OH_ROTATION_SUBGROUP[target])
    pairs = symmetry._target_rotations_in_oh(group)
    n_rot = len(pairs)
    mats = [_oh_double_irrep_matrices()[oh_double][i] for i, _gi in pairs]

    # the restriction is a projective representation: products land on ±member
    for A in mats:
        for B in mats:
            assert any(np.allclose(A @ B, sign * M, atol=1e-9)
                       for M in mats for sign in (1, -1)), \
                f"{oh_double} ⇓ {group.name} is not a projective rep"

    blocks = _commutant_blocks(mats)
    assert sorted(b.shape[1] for b in blocks) == sorted(expected_dims), (
        f"{oh_double} ⇓ {group.name}: blocks "
        f"{sorted(b.shape[1] for b in blocks)}, Butler says {sorted(expected_dims)}")

    # every block is irreducible: over the double group each unbarred element
    # pairs with a barred one of opposite character, so ⟨χ|χ⟩ = 2 Σ |χ|² / 2n
    for b in blocks:
        chi = np.array([np.trace(b.conj().T @ M @ b) for M in mats])
        norm = 2 * np.vdot(chi, chi).real / (2 * n_rot)
        assert abs(norm - 1.0) < 1e-9, (
            f"{oh_double} ⇓ {group.name}: a dim-{b.shape[1]} block has "
            f"⟨χ|χ⟩ = {norm:.6f}, so it is not irreducible")


def test_the_trigonal_double_group_has_one_dimensional_spinor_irreps():
    """Spelled out, because it is what makes step 2 more than a relabelling.

    Every D4h double irrep is 2-D, so the half-integer path has only ever seen
    dimension 2. D3's double group has two 1-D spinor irreps, a
    complex-conjugate pair, and they appear from J = 3/2 up.
    """
    d3 = point_group('D3', spinor=True)
    spinor = {d3.irrep_labels[i]: int(d3.irrep_dims[i])
              for i, is_spinor in enumerate(d3.spinor_irreps) if is_spinor}
    assert spinor == {'1E3/2': 1, '2E3/2': 1, 'E1/2': 2}, spinor

    d4 = point_group('D4', spinor=True)
    assert all(int(d4.irrep_dims[i]) == 2
               for i, is_spinor in enumerate(d4.spinor_irreps) if is_spinor)

    assert '1E3/2' in d3.branching(1.5) and d3.branching(1.5)['1E3/2'] == 1


DISK0 = Path(__file__).resolve().parents[2].parent / "ttmult" / "src" / "disk0"


@pytest.mark.skipif(not DISK0.exists(), reason=f"Butler tables not present at {DISK0}")
def test_the_butler_transcription_still_matches_disk0():
    """Drift guard on the transcription, not on the oracle.

    ``disk0`` is not part of this repository, so the table above is a
    transcription and the transcription is the weak link. When the Fortran tree
    is reachable, re-read the ``$F3/OD3`` block's ``BRANCHINGS`` line and
    require it to say what :data:`BUTLER_O_TO_D3` says.
    """
    text = DISK0.read_text(errors='replace')
    start = text.index('$F3/OD3')
    block = text[start:text.index('$F3/', start + 1)]
    branchings = block[block.index('BRANCHINGS') + len('BRANCHINGS'):
                       block.index('ENDBRANCHINGS')]

    parsed = {}
    for clause in branchings.split(';'):
        fields = clause.split()
        if not fields:
            continue
        parsed[fields[0]] = [f.lstrip('+') for f in fields[1:]]

    assert parsed == BUTLER_O_TO_D3, (
        f"disk0 says {parsed}, the transcription says {BUTLER_O_TO_D3}")
