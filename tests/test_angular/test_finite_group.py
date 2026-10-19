"""WP-B1: the numerical PointGroup against textbook group theory and the Oh path.

Nothing in :mod:`multitorch.angular.finite_group` is tabulated — a group is
given by generators and its elements, classes, characters and labels are all
computed — so the tests are the specification.

The oracles, strongest first:

* **Oh branching against** :func:`~multitorch.angular.point_group.oh_branching`,
  which is built from 24 hand-written rotation matrices and validated against
  the Fortran chain. The two constructions share no code, so agreement over
  J = 0..12 pins the element enumeration, the class decomposition, the
  character table and the branching formula at once.
* **The Oh double group against the standard Koster reduction** of D^(J) for
  half-integer J, and against the irrep names the package already uses
  (``E1/2``, ``E5/2``, ``G3/2``).
* **Textbook orders and class counts**, and Sigma d^2 = |G| with both
  character orthogonality relations.
* **Isomorphisms** O ~ Td, D4 ~ C4v, D3 ~ C3v: isomorphic groups must have the
  same multiset of characters, which catches a character table that is
  self-consistent but wrong for the group it claims.
* **Injected bugs**: a generator set that does not close, and a character table
  whose class-sum coefficients have been corrupted, must both be refused.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from multitorch.angular.finite_group import (
    Element, GENERATORS, INVERSION, PointGroup, point_group, quaternion,
    quaternion_product, quaternion_to_matrix, matrix_to_quaternion,
    wigner_from_quaternion,
)
from multitorch.angular.point_group import (
    OH_DOUBLE_IRREP_DIM, OH_IRREP_DIM, octahedral_rotations, oh_branching,
)
from multitorch.angular.symmetry import D4H_IRREP_DIM

# (name, |G|, number of classes) — standard tables
SINGLE = [
    ("C1", 1, 1), ("Ci", 2, 2), ("C2", 2, 2), ("C2h", 4, 4), ("C2v", 4, 4),
    ("D2", 4, 4), ("D2h", 8, 8), ("C3", 3, 3), ("C3v", 6, 3), ("D3", 6, 3),
    ("D3d", 12, 6), ("C4", 4, 4), ("C4h", 8, 8), ("C4v", 8, 5), ("D4", 8, 5),
    ("D4h", 16, 10), ("D2d", 8, 5), ("T", 12, 4), ("Th", 24, 8), ("Td", 24, 5),
    ("O", 24, 5), ("Oh", 48, 10),
]
SINGLE_IDS = [c[0] for c in SINGLE]

# the groups WP-B targets; their Mulliken labels must be the conventional ones
TARGETS = ["Oh", "O", "Td", "D4h", "D4", "C4v", "D3d", "D3", "C3v", "D2h", "D2", "C2v", "D2d"]


# ─────────────────────────────────────────────────────────────
# Quaternion algebra
# ─────────────────────────────────────────────────────────────

def test_quaternion_matrix_round_trip():
    """q -> R -> q is the identity up to the sign q ~ -q that SO(3) cannot see."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        axis = rng.normal(size=3)
        angle = rng.uniform(-math.pi, math.pi)
        q = quaternion(axis, angle)
        R = quaternion_to_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        back = matrix_to_quaternion(R)
        assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-9


def test_quaternion_product_matches_matrix_product():
    """The quaternion group law must be the rotation group law, in that order."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        p = quaternion(rng.normal(size=3), rng.uniform(-math.pi, math.pi))
        q = quaternion(rng.normal(size=3), rng.uniform(-math.pi, math.pi))
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion_product(p, q)),
            quaternion_to_matrix(p) @ quaternion_to_matrix(q), atol=1e-12)


def test_inversion_is_not_the_two_pi_quaternion():
    """Regression: inversion written as a 2pi rotation reports a 2pi angle.

    ``_improper(z, 1)`` builds quaternion (w = -1), the same O(3) operation as
    inversion but the SU(2) element -1. Everything that reads an element's
    rotation angle -- the Mulliken probes -- then misses it, which is how the
    first version of the labeller lost every g/u suffix.
    """
    assert INVERSION.parity == -1
    np.testing.assert_allclose(INVERSION.rotation, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(INVERSION.matrix, -np.eye(3), atol=1e-15)
    assert INVERSION.angle == pytest.approx(0.0, abs=1e-12)

    as_two_pi = Element(tuple(quaternion((0, 0, 1), 2 * math.pi)), -1)
    np.testing.assert_allclose(as_two_pi.matrix, INVERSION.matrix, atol=1e-12)
    assert as_two_pi.angle == pytest.approx(2 * math.pi, abs=1e-9)


# ─────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,order,n_classes", SINGLE, ids=SINGLE_IDS)
def test_single_group_order_and_classes(name, order, n_classes):
    g = point_group(name)
    assert g.order == order
    assert g.n_classes == n_classes
    assert len(g.irrep_labels) == n_classes          # #irreps = #classes
    assert int(np.sum(g.irrep_dims ** 2)) == order   # Burnside


@pytest.mark.parametrize("name,order,n_classes", SINGLE, ids=SINGLE_IDS)
def test_double_group_doubles_the_order(name, order, n_classes):
    """|G'| = 2|G| for every group, and the extra irreps are the spinor ones."""
    g = point_group(name, spinor=True)
    assert g.order == 2 * order
    assert int(np.sum(g.irrep_dims ** 2)) == 2 * order
    assert g.n_classes >= n_classes
    # every single-group irrep survives, and at least one spinor irrep appears
    assert sum(g.spinor_irreps) >= 1
    assert sum(not s for s in g.spinor_irreps) == n_classes


@pytest.mark.parametrize("name,order,n_classes", SINGLE, ids=SINGLE_IDS)
def test_characters_are_orthonormal(name, order, n_classes):
    """Both orthogonality relations, asserted from outside the constructor."""
    g = point_group(name)
    chi = g.character_table
    sizes = np.array([len(c) for c in g.classes], dtype=np.float64)
    np.testing.assert_allclose((chi * sizes) @ chi.conj().T / g.order,
                               np.eye(g.n_classes), atol=1e-8)
    cols = chi.conj().T @ chi / g.order
    np.testing.assert_allclose(cols * sizes[:, None], np.eye(g.n_classes), atol=1e-8)


@pytest.mark.parametrize("name", [s[0] for s in SINGLE])
@pytest.mark.parametrize("spinor", [False, True], ids=["single", "double"])
def test_first_irrep_is_totally_symmetric(name, spinor):
    """Widened from the 13 WP-B targets to all 22 groups, both covers, in B3.

    It had been asserted only on TARGETS, and T failed it: the three
    one-dimensional irreps of a complex-conjugate-pair family are labelled
    ``Aa``/``Ab``/``Ac`` in character-table order, the last sort tiebreaker is
    the label string, and the all-ones row was not the alphabetically first
    one. Found by WP-B3 asking every group for its totally symmetric subspace.
    """
    g = point_group(name, spinor=spinor)
    np.testing.assert_allclose(g.character_table[0], np.ones(g.n_classes), atol=1e-9)


def test_classes_partition_the_group():
    for name in TARGETS:
        g = point_group(name)
        members = sorted(i for c in g.classes for i in c)
        assert members == list(range(g.order))
        assert 0 in g.classes[0] and len(g.classes[0]) == 1   # identity alone


# ─────────────────────────────────────────────────────────────
# Against the hand-written Oh path
# ─────────────────────────────────────────────────────────────

def test_O_elements_are_the_octahedral_rotations():
    """The closure of two generators is exactly point_group's 24 matrices.

    Independent constructions: ``octahedral_rotations`` writes out the four
    body diagonals, three face axes and six edge axes by hand.
    """
    mine = [e.matrix for e in point_group("O").elements]
    theirs = octahedral_rotations()
    assert len(mine) == len(theirs) == 24
    for R in theirs:
        assert any(np.abs(R - M).max() < 1e-9 for M in mine), R


@pytest.mark.parametrize("J", list(range(0, 13)))
def test_Oh_integer_branching_matches_the_fortran_validated_path(J):
    """The headline oracle: Oh branching against ``oh_branching``, J = 0..12."""
    g = point_group("Oh")
    mine = {label[:-1]: n for label, n in g.branching(J, parity=+1).items()}
    theirs = {k: v for k, v in oh_branching(J).items() if v}
    assert mine == theirs


def test_Oh_single_group_labels_and_dims_match_the_package():
    g = point_group("Oh")
    dims = dict(zip(g.irrep_labels, (int(d) for d in g.irrep_dims)))
    for label, d in OH_IRREP_DIM.items():
        assert dims[f"{label}g"] == d
        assert dims[f"{label}u"] == d
    assert len(dims) == 2 * len(OH_IRREP_DIM)


def test_D4h_single_group_labels_and_dims_match_the_package():
    g = point_group("D4h")
    assert dict(zip(g.irrep_labels, (int(d) for d in g.irrep_dims))) == D4H_IRREP_DIM


def test_Oh_double_group_names_match_the_package():
    """The spinor irreps come out named E1/2, E5/2, G3/2 with the right dimensions.

    They are *derived*: a spinor irrep is one the 2pi rotation represents as
    -1, and it is named for the lowest half-integer J that contains it.
    """
    g = point_group("Oh", spinor=True)
    dims = dict(zip(g.irrep_labels, (int(d) for d in g.irrep_dims)))
    for label, d in OH_DOUBLE_IRREP_DIM.items():
        assert dims[f"{label}g"] == d, label
        assert dims[f"{label}u"] == d, label
    spinor = {lab for lab, s in zip(g.irrep_labels, g.spinor_irreps) if s}
    assert spinor == {f"{l}{p}" for l in OH_DOUBLE_IRREP_DIM for p in "gu"}


# Standard reduction of D^(J) in the Oh double group (Koster et al.):
# Gamma6 = E1/2, Gamma7 = E5/2, Gamma8 = G3/2.
OH_HALF_BRANCHING = {
    1: {"E1/2": 1},
    3: {"G3/2": 1},
    5: {"E5/2": 1, "G3/2": 1},
    7: {"E1/2": 1, "E5/2": 1, "G3/2": 1},
    9: {"E1/2": 1, "G3/2": 2},
    11: {"E1/2": 1, "E5/2": 1, "G3/2": 2},
}


@pytest.mark.parametrize("two_j", sorted(OH_HALF_BRANCHING))
def test_Oh_half_integer_branching_matches_the_standard_tables(two_j):
    g = point_group("Oh", spinor=True)
    for parity, suffix in ((+1, "g"), (-1, "u")):
        got = g.branching(two_j / 2.0, parity=parity)
        assert got == {f"{k}{suffix}": v for k, v in OH_HALF_BRANCHING[two_j].items()}


def test_half_integer_J_needs_the_double_group():
    with pytest.raises(ValueError, match="half-integer"):
        point_group("Oh").so3_character(2.5)


@pytest.mark.parametrize("name", TARGETS)
def test_branching_spans_the_whole_manifold(name):
    """Sigma mult * dim = 2J+1, and parity selects a single g/u set."""
    for spinor in (False, True):
        g = point_group(name, spinor=spinor)
        js = [0.5 * k for k in range(1, 14)] if spinor else list(range(0, 8))
        for J in js:
            for parity in (+1, -1):
                b = g.branching(J, parity=parity)
                dims = dict(zip(g.irrep_labels, (int(d) for d in g.irrep_dims)))
                assert sum(dims[k] * v for k, v in b.items()) == int(round(2 * J)) + 1
                is_spinor = dict(zip(g.irrep_labels, g.spinor_irreps))
                want = abs(J - round(J)) > 0.1
                assert all(is_spinor[k] == want for k in b), (name, J, b)


# ─────────────────────────────────────────────────────────────
# Isomorphisms
# ─────────────────────────────────────────────────────────────

def _character_multiset(g):
    """The characters as a sortable multiset of (dim, sorted class characters).

    Isomorphic groups have the same character table up to a permutation of
    classes and irreps, so comparing the sorted rows -- each row itself sorted
    -- is a relabelling-independent fingerprint.
    """
    rows = []
    for t in range(g.n_classes):
        vals = sorted((round(z.real, 6), round(z.imag, 6)) for z in g.character_table[t])
        rows.append((int(g.irrep_dims[t]), tuple(vals)))
    return sorted(rows)


@pytest.mark.parametrize("a,b", [("O", "Td"), ("D4", "C4v"), ("D3", "C3v"),
                                 ("D2", "C2v"), ("D4", "D2d")])
def test_isomorphic_groups_share_a_character_table(a, b):
    """O ~ Td ~ S4, D4 ~ C4v ~ D2d, D3 ~ C3v ~ S3, D2 ~ C2v.

    A character table can be internally consistent (orthonormal, Burnside) and
    still belong to the wrong group; this is the check that it does not.
    """
    ga, gb = point_group(a), point_group(b)
    assert ga.order == gb.order
    assert _character_multiset(ga) == _character_multiset(gb)


def test_cubic_and_axial_groups_are_not_confused():
    """A guard on the above: D4h and D2d both have order 8 subgroups but differ."""
    assert _character_multiset(point_group("D4h")) != _character_multiset(point_group("D2h"))
    assert _character_multiset(point_group("O")) != _character_multiset(point_group("D4h"))


# ─────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", TARGETS)
def test_target_group_labels_are_conventional(name):
    """Every group WP-B is for gets Mulliken labels with no synthetic suffix."""
    for spinor in (False, True):
        g = point_group(name, spinor=spinor)
        assert g.labels_are_conventional, (name, spinor, g.irrep_labels)
        assert len(set(g.irrep_labels)) == len(g.irrep_labels)


def test_complex_pair_families_are_flagged_not_faked():
    """C3, C4, C4h, T and Th have complex-conjugate pairs Mulliken primes.

    Those primes are not implemented, so the labels carry an a/b suffix and the
    group says so rather than claiming a convention it does not follow.
    """
    for name in ("C3", "C4", "C4h", "T", "Th"):
        g = point_group(name)
        assert not g.labels_are_conventional, name
        assert len(set(g.irrep_labels)) == len(g.irrep_labels)


def test_mulliken_letters_follow_the_dimension():
    for name in TARGETS:
        g = point_group(name)
        for label, d in zip(g.irrep_labels, g.irrep_dims):
            if int(d) == 1:
                assert label[0] in "AB", (name, label)
            elif int(d) == 2:
                assert label[0] == "E", (name, label)
            elif int(d) == 3:
                assert label[0] == "T", (name, label)


def test_A_and_B_split_on_the_principal_rotation():
    """B is the one-dimensional irrep odd under the principal axis, A the even one."""
    g = point_group("D4h")
    chi = dict(zip(g.irrep_labels, g.character_table))
    c4 = next(c for c, cl in enumerate(g.classes)
              if abs(g.elements[cl[0]].angle - math.pi / 2) < 1e-9
              and g.elements[cl[0]].parity > 0)
    for label in ("A1g", "A2g", "A1u", "A2u"):
        assert chi[label][c4].real == pytest.approx(1.0, abs=1e-9), label
    for label in ("B1g", "B2g", "B1u", "B2u"):
        assert chi[label][c4].real == pytest.approx(-1.0, abs=1e-9), label


# ─────────────────────────────────────────────────────────────
# Injected bugs
# ─────────────────────────────────────────────────────────────

def test_a_generator_set_that_does_not_close_is_refused():
    """A 2pi/5 rotation about an axis of a cubic group generates no finite group."""
    gens = (Element(tuple(quaternion((0, 0, 1), 2 * math.pi / 4)), +1),
            Element(tuple(quaternion((1, 1, 1), 2 * math.pi / 5)), +1))
    with pytest.raises(ValueError, match="finite group|exceeded"):
        PointGroup("bogus", gens)


@pytest.mark.parametrize("site", [(1, 1, 1), (4, 4, 4), (0, 1, 1), (4, 4, 0),
                                  (4, 5, 6), (2, 3, 4), (5, 5, 5), (9, 9, 9)])
def test_a_corrupted_class_algebra_is_refused(site):
    """Inject the bug the character guards exist for and watch each one fail.

    The class-sum (Dixon) construction is only as good as its coefficients, and
    a wrong one still gives a table of the right shape with the right number of
    rows. Each site below is a different failure mode, and two of them --
    ``(1,1,1)`` and ``(4,4,4)`` -- are the reason the tensor-product guard
    exists: measured on 2026-10-18, they produce a table that is wrong in every
    row while passing *both* orthogonality relations and Burnside. Only asking
    whether the rows are characters of this group at all rejects them.
    """
    broken = PointGroup(f"D4h-corrupt-{site}", GENERATORS["D4h"])
    corrupt = broken.class_multiplication_matrices.copy()
    corrupt[site] += 1
    broken.__dict__["class_multiplication_matrices"] = corrupt
    with pytest.raises(RuntimeError, match="orthonormal|orthogonal|sum of squares"
                                           "|not characters of the group|separate|not real"):
        _ = broken.character_table


def test_the_tensor_product_guard_accepts_every_real_group():
    """The new guard must not be so strict that it rejects a correct table."""
    for name, _, _ in SINGLE:
        for spinor in (False, True):
            g = point_group(name, spinor=spinor)
            chi = g.character_table
            sizes = np.array([len(c) for c in g.classes], dtype=np.float64)
            m = np.einsum("i,ui,vi,wi->uvw", sizes, chi, chi, chi.conj()) / g.order
            assert np.max(np.abs(m.imag)) < 1e-6, (name, spinor)
            assert np.max(np.abs(m.real - np.round(m.real))) < 1e-6, (name, spinor)
            assert np.min(m.real) > -1e-6, (name, spinor)


def test_wrong_so3_character_would_break_the_branching_guard():
    """branching() refuses a reduction that does not span the manifold.

    Injected by asking for the character of a parity the group cannot carry --
    here a J whose multiplicities would not be integers if the class
    representative were mismatched.
    """
    g = point_group("Oh")
    chi = g.so3_character(2, parity=+1)
    assert chi.shape == (g.n_classes,)
    # the identity class always carries 2J+1
    assert chi[0] == pytest.approx(5.0, abs=1e-12)
    # and a corrupted character is rejected rather than rounded
    import unittest.mock as mock
    bad = chi.copy()
    bad[0] += 1.0
    with mock.patch.object(PointGroup, "so3_character", return_value=bad):
        with pytest.raises(RuntimeError, match="non-integer|spans"):
            g.branching(2, parity=+1)


# ─────────────────────────────────────────────────────────────
# WP-B2: irrep matrices and O(3) -> G subduction
# ─────────────────────────────────────────────────────────────

def _isotypic(S, labels, dim, basis=None):
    """Per-irrep projector from a subduction, which is gauge-free."""
    from collections import defaultdict
    M = S if basis is None else basis @ S
    out = defaultdict(lambda: np.zeros((dim, dim), dtype=np.complex128))
    for k, (irrep, _copy, _mu) in enumerate(labels):
        v = M[:, k:k + 1]
        out[irrep] += v @ v.conj().T
    return out


@pytest.mark.parametrize("J", [1, 2, 3, 4, 5, 6])
def test_wigner_from_quaternion_matches_the_package_convention(J):
    """The exponential map, conjugated into the package's phase convention.

    Without the ``diag(exp(i pi m))`` similarity the two differ by O(1) in
    every element while having identical characters, which would make every
    subduction built here silently incomparable with the existing Oh path.
    """
    from multitorch.angular.point_group import wigner_D_matrix
    g = point_group("Oh")
    worst = max(np.abs(wigner_from_quaternion(J, e.quat) - wigner_D_matrix(J, e.rotation)).max()
                for e in g.elements)
    assert worst < 1e-12, worst


@pytest.mark.parametrize("two_j", [1, 3, 5])
def test_wigner_from_quaternion_is_a_spinor_representation(two_j):
    """Half-integer J: a homomorphism on the double group, and -1 on the 2pi rotation."""
    g = point_group("Oh", spinor=True)
    J = two_j / 2.0
    dim = two_j + 1
    for e in g.elements[:24]:
        D = wigner_from_quaternion(J, e.quat)
        np.testing.assert_allclose(D @ D.conj().T, np.eye(dim), atol=1e-12)
        other = g.elements[7]
        np.testing.assert_allclose(
            wigner_from_quaternion(J, (e * other).quat),
            D @ wigner_from_quaternion(J, other.quat), atol=1e-12)
    two_pi = Element((-1.0, 0.0, 0.0, 0.0), 1)
    np.testing.assert_allclose(wigner_from_quaternion(J, two_pi.quat),
                               -np.eye(dim), atol=1e-12)


@pytest.mark.parametrize("name", ["C2v", "D2h", "C4v", "D3d", "D4h", "Td", "O", "Oh"])
def test_irrep_matrices_are_unitary_homomorphisms_with_the_right_characters(name):
    """The construction verifies itself; this asserts it from outside too."""
    for spinor in (False, True):
        g = point_group(name, spinor=spinor)
        mats = g.irrep_matrices
        assert set(mats) == set(g.irrep_labels)
        for t, label in enumerate(g.irrep_labels):
            M = mats[label]
            d = int(g.irrep_dims[t])
            assert M.shape == (g.order, d, d), (name, label, M.shape)
            gram = np.einsum("gij,gkj->gik", M, M.conj())
            assert np.abs(gram - np.eye(d)).max() < 1e-8, (name, label)
            np.testing.assert_allclose(np.einsum("gii->g", M),
                                       g.character_table[t][g.class_of], atol=1e-7)


@pytest.mark.parametrize("J", list(range(0, 11)))
def test_Oh_subduction_matches_the_production_real_subduction(J):
    """The B2 oracle: isotypic subspaces against ``_real_subduction_matrix``.

    That is the function ``rac_generator`` actually builds the Oh RAC from, so
    it carries the Fortran validation. Projectors rather than columns, because
    the partner gauge is free and only the subspaces are physics.
    """
    from multitorch.angular.point_group import _c2r_unitary, _real_subduction_matrix
    g = point_group("Oh")
    S, labels = g.subduction(J, +1)
    U = _c2r_unitary(J) if J > 0 else np.ones((1, 1), dtype=np.complex128)
    mine = _isotypic(S, labels, 2 * J + 1, basis=U)
    for label, P in mine.items():
        B = _real_subduction_matrix(J, label[:-1])
        assert B.shape[1] == int(round(P.trace().real)), (J, label)
        np.testing.assert_allclose(P, B @ B.T, atol=1e-12, err_msg=f"J={J} {label}")


@pytest.mark.parametrize("two_j", [1, 3, 5, 7, 9, 11])
def test_Oh_half_integer_subduction_matches_the_production_double_group(two_j):
    """The same oracle for the double group, including the 4-dimensional G3/2."""
    from multitorch.angular.point_group import _complex_subduction_matrix_half_int
    g = point_group("Oh", spinor=True)
    J = two_j / 2.0
    S, labels = g.subduction(J, +1)
    mine = _isotypic(S, labels, two_j + 1)
    for label, P in mine.items():
        B = _complex_subduction_matrix_half_int(J, label[:-1])
        assert B.shape[1] == int(round(P.trace().real)), (J, label)
        np.testing.assert_allclose(P, B @ B.conj().T, atol=1e-12, err_msg=f"J={J} {label}")


@pytest.mark.parametrize("J", list(range(0, 9)))
def test_D4h_subduction_matches_the_production_partner_basis(J):
    """The D4h half of the B2 oracle, at the natural parity (-1)^J of D^J.

    ``d4h_partner_basis_per_J`` composes the Oh subduction with the hand-written
    Oh->D4h second step, which is the chain WP-B replaces; matching it means the
    direct O(3)->D4h projection needs no chain.
    """
    from multitorch.angular.point_group import _c2r_unitary
    from multitorch.angular.symmetry import d4h_partner_basis_per_J
    g = point_group("D4h")
    parity = +1 if J % 2 == 0 else -1
    S, labels = g.subduction(J, parity)
    U = _c2r_unitary(J) if J > 0 else np.ones((1, 1), dtype=np.complex128)
    mine = _isotypic(S, labels, 2 * J + 1, basis=U)
    for label, P in mine.items():
        B = d4h_partner_basis_per_J(J, label)
        assert B.shape[1] == int(round(P.trace().real)), (J, label, B.shape)
        np.testing.assert_allclose(P, B @ B.conj().T, atol=1e-12, err_msg=f"J={J} {label}")


@pytest.mark.parametrize("name", ["Oh", "D4h", "Td", "C4v", "D3d", "D2h", "C2v"])
def test_subduction_block_diagonalises_the_representation(name):
    """S^dag D(g) S is the direct sum of the irrep matrices, for every element.

    The strongest internal statement there is: it pins the multiplicities, the
    partner ordering and the gauge simultaneously, for groups that have no
    hand-written counterpart to compare against.
    """
    g = point_group(name)
    for J in range(0, 5):
        for parity in (+1, -1):
            S, labels = g.subduction(J, parity)
            dim = 2 * J + 1
            np.testing.assert_allclose(S.conj().T @ S, np.eye(dim), atol=1e-9)
            want = np.zeros((g.order, dim, dim), dtype=np.complex128)
            at = 0
            for irrep, _copy, mu in labels:
                if mu:
                    continue
                M = g.irrep_matrices[irrep]
                d = M.shape[1]
                want[:, at:at + d, at:at + d] = M
                at += d
            assert at == dim
            for gi, e in enumerate(g.elements):
                D = wigner_from_quaternion(J, e.quat) * (parity if e.parity < 0 else 1)
                np.testing.assert_allclose(S.conj().T @ D @ S, want[gi], atol=1e-8,
                                           err_msg=f"{name} J={J} parity={parity} element {gi}")


@pytest.mark.parametrize("name", ["Oh", "D4h", "Td", "D3d", "C2v"])
def test_subduction_agrees_with_the_gauge_free_character_projector(name):
    """Isotypic subspaces must equal the character projector, which has no gauge.

    ``P = (d/|G|) sum_g conj(chi(g)) D(g)`` involves no irrep matrices at all,
    so this separates "the subspaces are right" from "the partner basis inside
    them is the one we chose".
    """
    g = point_group(name)
    for J in range(0, 6):
        for parity in (+1, -1):
            S, labels = g.subduction(J, parity)
            dim = 2 * J + 1
            mine = _isotypic(S, labels, dim)
            D = np.stack([wigner_from_quaternion(J, e.quat) * (parity if e.parity < 0 else 1)
                          for e in g.elements])
            for t, label in enumerate(g.irrep_labels):
                if label not in mine:
                    continue
                d = int(g.irrep_dims[t])
                chi = g.character_table[t][g.class_of]
                P = np.einsum("g,gij->ij", chi.conj(), D) * (d / g.order)
                np.testing.assert_allclose(mine[label], P, atol=1e-10,
                                           err_msg=f"{name} J={J} {label}")


def test_dead_oh_subduction_matrix_disagrees_with_the_character_projector():
    """point_group.oh_subduction_matrix is unused and its degenerate blocks are wrong.

    Recorded as residual 38 rather than fixed: the function has **no call
    sites** in the package or the tests, and the production path goes through
    ``_real_subduction_matrix``, which uses the correct projector
    ``Gamma(R)[mu, nu].conj()`` where the dead one uses ``Gamma(R)[nu, mu]``.
    One-dimensional irreps survive the swap (the single diagonal element is the
    character), degenerate ones do not. This test pins the defect so that
    anyone who revives the function finds out, and so that WP-B2's oracle is
    not mistaken for it again.
    """
    from multitorch.angular.point_group import (
        OH_IRREP_DIM, _complex_to_real_matrix, _D_real_matrices,
        _oh_characters_at_rotations, octahedral_rotations, oh_subduction_matrix,
    )
    from collections import defaultdict

    rots = octahedral_rotations()
    bad = {}
    for J in (2, 3, 4):
        D_real = _D_real_matrices(J)
        S, labels = oh_subduction_matrix(J)
        theirs = defaultdict(lambda: np.zeros((2 * J + 1, 2 * J + 1)))
        for k, (irrep, _copy) in enumerate(labels):
            w = S[:, k:k + 1]
            theirs[irrep] += w @ w.T
        for irrep, P_theirs in theirs.items():
            chi = _oh_characters_at_rotations(rots, irrep)
            P = sum(c * D for c, D in zip(chi, D_real)) * (OH_IRREP_DIM[irrep] / 24.0)
            # the character projector is the unambiguous answer: idempotent,
            # right rank, and the projectors sum to the identity
            np.testing.assert_allclose(P @ P, P, atol=1e-10)
            err = float(np.abs(P - P_theirs).max())
            if OH_IRREP_DIM[irrep] > 1:
                bad[(J, irrep)] = err
            else:
                assert err < 1e-10, (J, irrep, err)
    assert bad, "oh_subduction_matrix now agrees; residual 38 may be fixed"
    assert min(bad.values()) > 0.1, bad
