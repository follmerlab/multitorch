"""Symmetry-adapted crystal field derived from the group (WP-B3).

Oracles, none of them code-vs-code in the sense that matters:

1. **Fortran, through the live path.** The derived operator vectors must equal
   ``rac_generator._d4h_operator_vector_butler``'s output for TENDQ, DT and
   DS. That function is the single source of crystal-field operator vectors for
   *both* groups and *both* parities of J — ``_make_oh_op_adds``,
   ``_make_oh_op_adds_half``, ``_make_d4h_op_adds`` and the double-group
   emitters all call it — and its numbers are what make the D4h store match the
   ``nid8ct`` fixture and the Oh stores match theirs. It is kept as a reference
   implementation precisely so this comparison survives; see
   ``test_the_butler_route_implementation_is_kept_as_the_oracle``.

2. **Analytic.** Ballhausen's one-electron d energies, which are a textbook
   convention independent of this package: E(xy) = -4Dq + 2Ds - Dt, E(xz) =
   E(yz) = -4Dq - Ds + 4Dt, E(z2) = 6Dq - 2Ds - 6Dt, E(x2-y2) = 6Dq + 2Ds - Dt.

3. **Structural.** The number of free parameters is derived, never tabulated,
   and must come out at the conventional count for every group: 1 for the cubic
   groups, 3 for tetragonal and trigonal, 5 for orthorhombic.
"""
from __future__ import annotations

import numpy as np
import pytest

from multitorch.angular import crystal_field as cf
from multitorch.angular.crystal_field import (
    CF_RANKS, D_ORBITAL_ORDER, ONE_ELECTRON_PATTERNS, cf_invariant_vectors,
    cf_operator_from_pattern, cf_operator_vector, cf_parameter_count)
from multitorch.angular.finite_group import point_group, wigner_from_quaternion
from multitorch.angular.rac_generator import (
    _d4h_operator_vector_butler, _operator_real_matrix)

#: (group, k=2 count, k=4 count). The conventional crystal-field parameter
#: counts: cubic has 10Dq alone; tetragonal and trigonal have three
#: ((10Dq, Ds, Dt) and (10Dq, Dsigma, Dtau)); orthorhombic has five.
COUNTS = [
    ("Oh", 0, 1), ("O", 0, 1), ("Td", 0, 1), ("T", 0, 1),
    ("D4h", 1, 2), ("D4", 1, 2), ("C4v", 1, 2), ("D2d", 1, 2),
    ("D3d", 1, 2), ("D3", 1, 2), ("C3v", 1, 2),
    ("D2h", 2, 3), ("D2", 2, 3), ("C2v", 2, 3),
]

SQRT5 = np.sqrt(5.0)


# ─────────────────────────────────────────────────────────────
# Structural: the parameter count is derived
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group,n2,n4", COUNTS, ids=[c[0] for c in COUNTS])
def test_parameter_counts_are_the_conventional_ones(group, n2, n4):
    """Nothing is tabulated: these come out of the subduction."""
    assert cf_parameter_count(group) == {2: n2, 4: n4}


@pytest.mark.parametrize("group", [c[0] for c in COUNTS])
@pytest.mark.parametrize("k", CF_RANKS)
def test_invariant_vectors_are_invariant_under_every_element(group, k):
    """``D^(k)(g) B = B`` for every g — the property that makes these allowed."""
    V = cf_invariant_vectors(group, k)
    g = point_group(group)
    for e in g.elements:
        D = wigner_from_quaternion(float(k), e.quat)
        assert np.allclose(D @ V, V, atol=1e-11), (group, k, e)


@pytest.mark.parametrize("group", [c[0] for c in COUNTS])
@pytest.mark.parametrize("k", CF_RANKS)
def test_every_invariant_is_traceless_on_the_d_shell(group, k):
    """k > 0 carries no configuration average; that constant is the k = 0 term."""
    V = cf_invariant_vectors(group, k)
    for i in range(V.shape[1]):
        M = _operator_real_matrix(2.0, 2.0, k, V[:, i])
        assert abs(np.trace(M)) < 1e-11, (group, k, i, np.trace(M))


def test_the_crystal_field_is_gerade_so_odd_ranks_are_refused():
    with pytest.raises(ValueError, match="gerade"):
        cf_invariant_vectors("Oh", 3)


# ─────────────────────────────────────────────────────────────
# Oracle 1: the Fortran-validated hand-coded vectors
# ─────────────────────────────────────────────────────────────

#: (group, named operator here, operator name in the Butler implementation).
#: Oh's 10DQ and D4h's TENDQ are the same cubic operator, and the Oh emitters
#: do in fact call the Butler routine with ``'TENDQ'``.
HAND_CODED = [("D4h", "TENDQ", "TENDQ"), ("D4h", "DS", "DS"),
              ("D4h", "DT", "DT"), ("Oh", "10DQ", "TENDQ")]


@pytest.mark.parametrize("group,operator,butler", HAND_CODED,
                         ids=[f"{g}.{o}" for g, o, _ in HAND_CODED])
def test_derived_operators_equal_the_fortran_validated_hand_coded_ones(group, operator, butler):
    """The derivation reproduces the Butler table *and* both of its gauge pins.

    Measured: 4.5e-15 (TENDQ), 7.1e-15 (DS), 3.9e-14 (DT) on vectors of norm
    3.29, 8.37 and 25.10, so ~1.6e-15 relative.
    """
    k, derived = cf_operator_vector(group, operator)
    hand = _d4h_operator_vector_butler(butler, k)
    assert derived.shape == hand.shape
    err = float(np.max(np.abs(derived - hand)))
    assert err < 1e-12 * max(1.0, float(np.linalg.norm(hand))), (group, operator, err)


def test_the_butler_route_implementation_is_kept_as_the_oracle():
    """``_d4h_operator_vector_butler`` and its table must stay reachable.

    It is no longer the production route, but it is the only *independent*
    construction of these vectors — Butler-weighted Oh-route sums against a
    projection onto the invariant subspace, sharing no code — and it is the one
    whose numbers were validated against the Fortran stores. Deleting it would
    delete the oracle, which is how B2's named oracle came to be dead and wrong
    (plan residual 38).
    """
    from multitorch.angular.symmetry import D4H_BRANCHES_BY_OPERATOR

    assert set(D4H_BRANCHES_BY_OPERATOR) == {"TENDQ", "DT", "DS"}
    for op, rank in (("TENDQ", 4), ("DT", 4), ("DS", 2)):
        v = _d4h_operator_vector_butler(op, rank)
        assert v.shape == (2 * rank + 1,) and np.any(np.abs(v) > 1e-9)


# ─────────────────────────────────────────────────────────────
# Oracle 2: Ballhausen's one-electron energies
# ─────────────────────────────────────────────────────────────

def test_d_orbital_order_is_what_the_cubic_operator_says():
    """Pin :data:`D_ORBITAL_ORDER` by measurement, not by trusting a comment.

    The cubic operator splits the shell into eg (+6Dq, two orbitals) and t2g
    (-4Dq, three), so the pattern itself says which index is which, and the
    ratio of the two levels must be exactly -3/2.
    """
    k, v = cf_operator_vector("Oh", "10DQ")
    d = np.diag(_operator_real_matrix(2.0, 2.0, k, v))
    eg = [i for i, x in enumerate(d) if x > 0]
    t2g = [i for i, x in enumerate(d) if x < 0]
    assert [D_ORBITAL_ORDER[i] for i in eg] == ["z2", "x2-y2"]
    assert [D_ORBITAL_ORDER[i] for i in t2g] == ["xy", "yz", "xz"]
    assert d[eg[0]] / d[t2g[0]] == pytest.approx(-1.5, abs=1e-12)


@pytest.mark.parametrize("group,operator", [(g, o) for g in ONE_ELECTRON_PATTERNS
                                            for o in ONE_ELECTRON_PATTERNS[g]])
def test_one_electron_matrix_is_sqrt_2l_plus_1_times_the_ballhausen_pattern(group, operator):
    """The analytic oracle, and the single normalisation constant with it.

    Read **in the frame the convention is written in** (residual 53): an axial
    pattern of a group whose principal axis is off z -- D3d's ``DSIGMA`` and
    ``DTAU`` -- is diagonal in the axial frame and emphatically not in the
    registered one, where the trigonal field mixes the d orbitals. For every
    group whose axis is already z the rotation is the exact identity, so this is
    the same assertion it always was for Oh and D4h.
    """
    k, pattern, frame = ONE_ELECTRON_PATTERNS[group][operator]
    v = cf_operator_vector(group, operator)[1]
    M = _operator_real_matrix(2.0, 2.0, k, v)
    if frame == "axial":
        D = np.asarray(cf.axial_frame_rotation(group))
        M = D @ M.astype(complex) @ D.conj().T
        assert np.max(np.abs(M.imag)) < 1e-11
        M = M.real
    assert np.max(np.abs(M - np.diag(np.diag(M)))) < 1e-11, "an A1 operator is diagonal here"
    assert np.allclose(np.diag(M), SQRT5 * np.asarray(pattern), atol=1e-11)


def test_the_oh_eg_t2g_splitting_is_exactly_ten_dq():
    """E(eg) - E(t2g) = 10Dq, from the operator alone."""
    k, v = cf_operator_vector("Oh", "10DQ")
    d = np.diag(_operator_real_matrix(2.0, 2.0, k, v)) / SQRT5   # back to Ballhausen units
    assert d[D_ORBITAL_ORDER.index("z2")] - d[D_ORBITAL_ORDER.index("xy")] == pytest.approx(1.0, abs=1e-12)


def test_ballhausen_superposition_reproduces_the_textbook_level_formulae():
    """A D4h field at (10Dq, Ds, Dt) gives Ballhausen's four one-electron levels.

    This is the whole named-parameter adapter, checked as a formula rather than
    operator by operator: the three operators are summed with arbitrary
    coefficients and the result compared with the textbook expressions.
    """
    tendq, ds, dt = 1.3, 0.11, 0.047
    dq = tendq / 10.0
    total = np.zeros((5, 5))
    for operator, amount in (("TENDQ", tendq), ("DS", ds), ("DT", dt)):
        k, v = cf_operator_vector("D4h", operator)
        total += amount * _operator_real_matrix(2.0, 2.0, k, v)
    got = {name: total[i, i] / SQRT5 for i, name in enumerate(D_ORBITAL_ORDER)}
    want = {"xy": -4 * dq + 2 * ds - dt, "yz": -4 * dq - ds + 4 * dt,
            "xz": -4 * dq - ds + 4 * dt, "z2": 6 * dq - 2 * ds - 6 * dt,
            "x2-y2": 6 * dq + 2 * ds - dt}
    for name in D_ORBITAL_ORDER:
        assert got[name] == pytest.approx(want[name], abs=1e-12), name


# ─────────────────────────────────────────────────────────────
# Refusals: a convention the symmetry forbids
# ─────────────────────────────────────────────────────────────

def test_oh_has_no_rank_two_crystal_field_parameter():
    """There is no Ds in Oh, and asking is an error rather than a zero."""
    assert cf_invariant_vectors("Oh", 2).shape[1] == 0
    with pytest.raises(ValueError, match="no totally symmetric invariant"):
        cf_operator_from_pattern("Oh", 2, ONE_ELECTRON_PATTERNS["D4h"]["DS"][1])


def test_a_pattern_outside_the_groups_invariants_is_refused():
    """Oh *has* a rank-4 invariant, but the tetragonal Dt pattern is not it."""
    with pytest.raises(ValueError, match="not a rank-4 invariant"):
        cf_operator_from_pattern("Oh", 4, ONE_ELECTRON_PATTERNS["D4h"]["DT"][1])


def test_unknown_group_and_operator_are_refused_with_a_useful_message():
    """D3d used to be the example here; residual 53 gave it a convention.

    D2h is the one that still has none, and for the reason D3d no longer does:
    its five-parameter set has no published one-electron table in this package's
    hands. Replacing the example rather than deleting the test keeps the refusal
    exercised, and the swap is the record that the D3d refusal was lifted on
    purpose rather than by a lookup drifting into the table.
    """
    with pytest.raises(ValueError, match="no named crystal-field convention"):
        cf_operator_vector("D2h", "DSIGMA")
    with pytest.raises(ValueError, match="unknown crystal-field operator"):
        cf_operator_vector("D4h", "DU")
    with pytest.raises(ValueError, match="unknown frame"):
        cf_operator_from_pattern("D4h", 2, ONE_ELECTRON_PATTERNS["D4h"]["DS"][1],
                                 frame="trigonal")


# ─────────────────────────────────────────────────────────────
# Injected bugs: each oracle must actually fail
# ─────────────────────────────────────────────────────────────

def test_dropping_the_unit_tensor_normalisation_breaks_the_hand_coded_agreement(monkeypatch):
    """Inject the missing ``sqrt(2l+1)`` and watch oracle 1 fail.

    Done by scaling the convention by ``1/sqrt(2l+1)``, which is exactly what
    "the normalisation was dropped" means for the target the solve matches. The
    result stays *self*-consistent — the module's own checks all pass — and is
    wrong only against the Butler route, which is the point of keeping that
    route as an oracle. The normalisation is the module's one constant and was
    measured rather than derived, so it needs this.
    """
    patched = dict(ONE_ELECTRON_PATTERNS)
    patched["D4h"] = {op: (k, tuple(x / SQRT5 for x in pattern), frame)
                      for op, (k, pattern, frame) in ONE_ELECTRON_PATTERNS["D4h"].items()}
    monkeypatch.setattr(cf, "ONE_ELECTRON_PATTERNS", patched)

    k, derived = cf_operator_vector("D4h", "TENDQ")
    hand = _d4h_operator_vector_butler("TENDQ", k)
    assert float(np.max(np.abs(derived - hand))) > 1e-3, "the normalisation went unnoticed"
    # and it is wrong by exactly that factor, not by something random
    assert np.allclose(derived * SQRT5, hand, atol=1e-11)


def test_swapping_two_conventions_breaks_the_analytic_oracle():
    """Inject a wrong pattern and watch oracle 2 fail: Dt solved as if it were Ds."""
    k_ds, pattern_ds, _ = ONE_ELECTRON_PATTERNS["D4h"]["DS"]
    k_dt, pattern_dt, _ = ONE_ELECTRON_PATTERNS["D4h"]["DT"]
    wrong = cf_operator_from_pattern("D4h", k_dt, pattern_dt)
    M = _operator_real_matrix(2.0, 2.0, k_dt, wrong)
    # It solves, because Dt is a genuine rank-4 invariant, but what comes back is
    # the Dt pattern and not the Ds one, which is what the oracle would catch.
    assert np.allclose(np.diag(M), SQRT5 * np.asarray(pattern_dt), atol=1e-11)
    assert np.max(np.abs(np.diag(M) - SQRT5 * np.asarray(pattern_ds))) > 1.0


def test_a_non_invariant_vector_is_caught_by_the_invariance_check():
    """Feed a column from a *non*-totally-symmetric irrep and watch it be refused."""
    g = point_group("D4h")
    S, labels = g.subduction(4, +1)
    totally_symmetric = g.irrep_labels[0]
    other = [i for i, (irrep, _c, _p) in enumerate(labels) if irrep != totally_symmetric]
    assert other, "D4h J=4 must contain more than the totally symmetric irrep"
    bad = S[:, other[0]]
    moved = [e for e in g.elements
             if not np.allclose(wigner_from_quaternion(4.0, e.quat) @ bad, bad, atol=1e-9)]
    assert moved, "a non-totally-symmetric vector must be moved by some element"


def _solve_in_basis(k, V, pattern, l=2):
    """Re-run the pattern solve against a supplied invariant basis."""
    from multitorch.angular.point_group import _c2r_unitary

    U = _c2r_unitary(l)
    target = U.conj().T @ (SQRT5 * np.diag(np.asarray(pattern))).astype(complex) @ U
    basis = np.column_stack([cf._one_electron_matrix_complex(k, V[:, i], l).ravel()
                             for i in range(V.shape[1])])
    coeffs, *_ = np.linalg.lstsq(basis, target.ravel(), rcond=None)
    return V @ coeffs, float(np.max(np.abs(basis @ coeffs - target.ravel())))


def test_the_named_operator_is_independent_of_the_invariant_bases_gauge():
    """Any unitary change of basis inside the invariant subspace must be invisible.

    This is the property that retires the two hand-placed gauge pins of the
    Butler route (plan residual 21 and the Ballhausen sign pin): the solve fixes
    direction, scale and sign, so whatever gauge ``eigh`` returned cannot show
    through.

    **Complex phases are the cases that matter, and the first version of this
    test missed them** by using a real rotation. The invariant basis comes out
    of ``eigh`` essentially real when left alone (max imaginary part 4.4e-16),
    so a real-only check passes against a solve that is in fact phase-sensitive
    — which is what the LAPACK scramble harness then caught in four integration
    tests. Measured here at <= 4e-14 for phases, mixed phases and random
    complex unitary mixes alike.
    """
    k, pattern, _ = ONE_ELECTRON_PATTERNS["D4h"]["DT"]
    want = cf_operator_vector("D4h", "DT")[1]
    V = np.array(cf_invariant_vectors("D4h", k))
    assert V.shape[1] == 2

    rng = np.random.default_rng(20261019)
    gauges = [np.eye(2, dtype=complex),
              np.diag([1j, 1.0 + 0j]),
              np.diag([np.exp(0.7j), 1.0 + 0j]),
              np.diag([np.exp(2.1j), np.exp(-1.3j)])]
    for _ in range(4):
        Q, _r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        gauges.append(Q)

    for i, M in enumerate(gauges):
        got, residual = _solve_in_basis(k, V @ M, pattern)
        assert residual < 1e-11, (i, residual)
        assert np.allclose(got, want, atol=1e-11), (i, float(np.max(np.abs(got - want))))


def test_solving_in_the_real_basis_is_phase_sensitive_and_can_fail_silently():
    """Inject the bug the complex solve exists to prevent, and watch it bite.

    ``_operator_real_matrix`` keeps only the real part, so a complex phase on an
    invariant basis vector destroys the information the solve needs. The
    dangerous part is not that it is wrong but that it can be wrong *quietly*:
    at a phase of ``exp(0.7i)`` the answer is off by ~21 in a vector of norm 25
    while the solve's own residual stays at ~5e-15.
    """
    k, pattern, _ = ONE_ELECTRON_PATTERNS["D4h"]["DT"]
    want = cf_operator_vector("D4h", "DT")[1]
    V = np.array(cf_invariant_vectors("D4h", k))
    target = (SQRT5 * np.diag(np.asarray(pattern))).ravel()

    def real_basis_solve(Vb):
        basis = np.column_stack([_operator_real_matrix(2.0, 2.0, k, Vb[:, i]).ravel()
                                 for i in range(Vb.shape[1])])
        coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
        return Vb @ coeffs.astype(np.complex128), float(np.max(np.abs(basis @ coeffs - target)))

    # unphased, the real solve happens to agree -- which is why it survived
    got, _r = real_basis_solve(V)
    assert np.allclose(got, want, atol=1e-11)

    phased = V.copy()
    phased[:, 0] = phased[:, 0] * np.exp(0.7j)
    got, residual = real_basis_solve(phased)
    assert float(np.max(np.abs(got - want))) > 1.0, "the real-basis solve was not phase-sensitive"
    assert residual < 1e-11, "the silent-failure half of this is the point"

    # the complex solve, same gauge, is unmoved
    got_complex, _r = _solve_in_basis(k, phased, pattern)
    assert np.allclose(got_complex, want, atol=1e-11)


# ─────────────────────────────────────────────────────────────
# Group-theory cross-checks, independent of this implementation
# ─────────────────────────────────────────────────────────────

#: Families that must share an invariant space on a d shell. A rank-k operator
#: with even k cannot see inversion (``D^(k)(i) = +1``), so groups differing
#: only by improper elements are indistinguishable here: Oh/O/Td/T all act the
#: same way on even-rank tensors, and so do D4h/D4/C4v/D2d.
FAMILIES = [
    ("cubic", ["Oh", "O", "Td", "T"]),
    ("tetragonal", ["D4h", "D4", "C4v", "D2d"]),
    ("trigonal", ["D3d", "D3", "C3v"]),
    ("trigonal[z]", ["D3d[z]", "D3[z]", "C3v[z]"]),
    ("orthorhombic", ["D2h", "D2", "C2v"]),
]

#: Registered groups that are genuinely subgroups of the standard Oh, so the
#: cubic field is one of their invariants. **The trigonal groups joined this
#: list in WP-B5-8** — they were absent while they were registered in the z
#: setting, which is the whole of residual 40. Widening the list is the point of
#: that increment, not an incidental edit: the property is what makes "10Dq plus
#: a distortion" meaningful, and a property asserted only over the groups that
#: already satisfied it is not asserted (the handoff's lesson 4, which is how
#: WP-B3 found `test_first_irrep_is_totally_symmetric` false for T).
#: The ``[z]`` variants are excluded by construction — see
#: :func:`test_the_z_setting_trigonal_groups_are_not_octahedral_subgroups`.
OCTAHEDRAL_SUBGROUPS = ["Oh", "O", "Td", "T", "D4h", "D4", "C4v", "D2d",
                        "D3d", "D3", "C3v",
                        "D2h", "D2", "C2v"]


@pytest.mark.parametrize("family,groups", FAMILIES, ids=[f[0] for f in FAMILIES])
@pytest.mark.parametrize("k", CF_RANKS)
def test_groups_differing_only_by_improper_elements_share_their_invariants(family, groups, k):
    """Measured at <= 6.3e-16. Compared as projectors, which is gauge-free."""
    def projector(name):
        V = cf_invariant_vectors(name, k)
        return V @ V.conj().T if V.shape[1] else np.zeros((2 * k + 1, 2 * k + 1), complex)

    ref = projector(groups[0])
    for name in groups[1:]:
        assert np.allclose(projector(name), ref, atol=1e-12), (family, k, name)


@pytest.mark.parametrize("group", OCTAHEDRAL_SUBGROUPS)
def test_the_cubic_invariant_lies_inside_every_octahedral_subgroups_invariants(group):
    """A subgroup of Oh must admit the cubic field: 10Dq survives the descent.

    This is the statement that makes "10Dq plus a distortion" a meaningful
    parameterisation, and it is pure group theory, so it catches a wrong
    invariant subspace without reference to any number in this package.
    Measured residuals are <= 6.5e-16.
    """
    cubic = cf_invariant_vectors("Oh", 4)[:, 0]
    V = cf_invariant_vectors(group, 4)
    P = V @ V.conj().T
    assert np.linalg.norm(cubic - P @ cubic) < 1e-12, group


@pytest.mark.parametrize("group", ["D3d", "D3", "C3v"])
def test_the_trigonal_groups_are_registered_in_the_octahedral_setting(group):
    """WP-B5-8 settled residual 40: C3 along [111], not along z.

    Every trigonal group in this package is reached by distorting an
    octahedron, and the trigonal subgroup of an octahedron has its C3 along a
    body diagonal. Registered that way each is a genuine subgroup of the
    standard Oh — **0 elements outside** — and the cubic invariant is one of
    its invariants, which is what makes "10Dq plus a trigonal distortion" a
    meaningful parameterisation at all.

    The crystallographic z setting is still reachable as ``D3d[z]`` etc. and is
    checked by :func:`test_the_z_setting_trigonal_groups_are_not_octahedral_subgroups`,
    which keeps residual 40's own measurement as the contrast.
    """
    from multitorch.angular.finite_group import point_group

    oh_mats = np.array([e.matrix for e in point_group("Oh").elements])
    g = point_group(group)
    outside = sum(
        np.min(np.abs(oh_mats - e.matrix).reshape(len(oh_mats), -1).max(axis=1)) > 1e-9
        for e in g.elements)
    assert outside == 0, f"{group} must be an Oh subgroup in the octahedral setting"

    cubic = cf_invariant_vectors("Oh", 4)[:, 0]
    V = cf_invariant_vectors(group, 4)
    P = V @ V.conj().T
    assert np.linalg.norm(cubic - P @ cubic) < 1e-12


@pytest.mark.parametrize("group,outside,order", [
    ("D3d[z]", 8, 12), ("D3[z]", 4, 6), ("C3v[z]", 4, 6),
])
def test_the_z_setting_trigonal_groups_are_not_octahedral_subgroups(group, outside, order):
    """Residual 40's measurement, kept as the contrast that justifies B5-8.

    These are the crystallographic standard setting and correct for a genuinely
    trigonal site. They are *not* octahedral subgroups, and the cubic invariant
    is correspondingly not one of their invariants — residual 0.6455, the same
    for all three because they share the axis. Keeping them reachable means the
    z setting is a choice a user can make rather than a thing the package lost,
    and keeping this test means the 0.6455 cannot quietly become something else.
    """
    from multitorch.angular.finite_group import point_group

    oh_mats = np.array([e.matrix for e in point_group("Oh").elements])
    g = point_group(group)
    assert len(g.elements) == order
    n_outside = sum(
        np.min(np.abs(oh_mats - e.matrix).reshape(len(oh_mats), -1).max(axis=1)) > 1e-9
        for e in g.elements)
    assert n_outside == outside

    cubic = cf_invariant_vectors("Oh", 4)[:, 0]
    V = cf_invariant_vectors(group, 4)
    P = V @ V.conj().T
    assert np.linalg.norm(cubic - P @ cubic) == pytest.approx(0.6455, abs=1e-3)


@pytest.mark.parametrize("pair", [("D3d", "D3d[z]"), ("D3", "D3[z]"), ("C3v", "C3v[z]")])
def test_the_two_settings_are_the_same_abstract_group(pair):
    """Re-orienting a group changes where it sits, not what it is.

    Same order, same character table up to class order, same irrep labels, and
    conventional labels in both — so B5-8 cost nothing in labelling, which is
    why ``PRINCIPAL_AXIS`` had to be filled in alongside the generators rather
    than instead of them. The free parameter count is orientation-independent
    too, which residual 40 already noted and this now asserts.
    """
    from multitorch.angular.crystal_field import cf_parameter_count
    from multitorch.angular.finite_group import point_group

    oct_g, z_g = point_group(pair[0]), point_group(pair[1])
    assert len(oct_g.elements) == len(z_g.elements)
    assert oct_g.irrep_labels == z_g.irrep_labels
    assert oct_g.labels_are_conventional and z_g.labels_are_conventional
    assert cf_parameter_count(pair[0]) == cf_parameter_count(pair[1])


def test_principal_axis_alone_does_not_reorient_a_group():
    """The correction to residual 40's framing, as a test.

    Residual 40 read as though filling ``PRINCIPAL_AXIS`` were the fix. It is
    half of it: the map only tells the Mulliken rules where the axis is, while
    the **generators** carry the orientation. A z-setting group declared to have
    a [111] principal axis is still not an Oh subgroup.
    """
    from multitorch.angular.finite_group import (INVERSION, PointGroup, _rot,
                                                 _X, _Z, point_group)

    oh_mats = np.array([e.matrix for e in point_group("Oh").elements])
    mislabelled = PointGroup(
        "D3d[z, claiming 111]",
        (_rot(_Z, 3), _rot(_X, 2), INVERSION),
        principal_axis=(1.0, 1.0, 1.0),
    )
    outside = sum(
        np.min(np.abs(oh_mats - e.matrix).reshape(len(oh_mats), -1).max(axis=1)) > 1e-9
        for e in mislabelled.elements)
    assert outside == 8, "declaring the axis cannot move the elements"


def test_the_cached_invariant_basis_cannot_be_mutated():
    """``cf_invariant_vectors`` is lru_cached, so an in-place edit by one caller
    would silently change what every later caller gets."""
    V = cf_invariant_vectors("D4h", 4)
    with pytest.raises(ValueError, match="read-only"):
        V[0, 0] = 99.0
