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
