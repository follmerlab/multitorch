"""The descent limit to Oh, resolved by route (WP-B, residual 49's rung 4).

Oh is the only one of the 14 WP-B target groups with a full Fortran
validation. Residual 49's ladder says that validation **transfers down the
whole chain**, because every Oh subgroup admits Oh's own cubic invariant: at
zero distortion a lower group's crystal field is not merely close to Oh's, it
is built from the same operator. ``test_crystal_field.py`` already asserts the
*containment* (the cubic invariant lies inside each group's invariant space).
This module makes the descent limit itself exact, and resolves it by the Oh
irrep each crystal-field parameter descends through.

Why route-resolved. ``scripts/derive_butler_routes.py`` (residual 50) derives
each group's ``BRANCH`` routes from Butler (1981), and those routes name the
intermediate Oh irrep: D4h's two rank-4 routes pass through Oh **A1** and Oh
**E**, D3d's through Oh **A1** and Oh **T2**. The script's gate validates its
parser on the *tetragonal* chain only -- it reproduces ``pyctm/write_RAC.py``'s
hard-coded Oh and D4h routes, which is all the pre-existing oracle covers. The
trigonal routes, which is what the exercise was for, were taken on the parser's
word. Here they are derived independently and compared.

**The two sides, which are genuinely two things.**

* Butler (1981), transcribed in :data:`BUTLER_OH_ROUTES` -- a published table,
  read from ``ttmult/src/disk0`` (the file ``ttrac`` itself reads) by
  ``scripts/derive_butler_routes.py``. Provenance and the exact blocks are in
  that table's docstring.
* :func:`oh_route_multiplicities` -- the Oh **isotypic decomposition** of the
  group's totally symmetric subspace of ``D^(k)``. The Oh isotypic projector
  ``P_Gamma`` is central in the image of the Oh group algebra, so it commutes
  with every ``D^(k)(h)``, ``h`` in G, for any subgroup G of Oh. The
  G-invariant subspace therefore splits orthogonally into its Oh-irrep pieces,
  and the dimension of each piece is the number of Butler routes through that
  Oh irrep. Nothing tabulated, and no route table consulted.

That the commuting argument needs ``G <= Oh`` is not a technicality: it is why
:func:`test_the_z_setting_trigonal_groups_break_the_route_decomposition` exists.

**What each rung here catches, measured by injection rather than argued.** Four
wrong invariant subspaces were substituted for real ones and every test rerun;
the results are in each test's docstring, and the summary is:

================================  ==========  ==========  ==========
injection                         routes      Oh limit    splitting
================================  ==========  ==========  ==========
D3d <- D3d[z] (wrong setting)     rejects     rejects     **passes**
D3d <- D4h (wrong family)         rejects     **passes**  rejects
T <- D3d (distortion in a cubic)  rejects     --          rejects
================================  ==========  ==========  ==========

So the route test is the only one of the three that sees a wrong setting *and*
a wrong family, which is the reason this module exists; the other two are
checks on different things and their blind spots are recorded where they are,
not left to be discovered later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.crystal_field import (
    CF_RANKS, D_ORBITAL_ORDER, ONE_ELECTRON_PATTERNS, _one_electron_matrix,
    cf_invariant_vectors, cf_operator_from_pattern, cf_operator_vector)
from multitorch.angular.finite_group import point_group, wigner_from_quaternion

#: The 14 WP-B target groups, in the B-status table's order. Every one is a
#: subgroup of the standard Oh in the setting this package registers it in --
#: ``test_crystal_field.py`` measures 0 elements outside Oh for all of them.
TARGET_GROUPS = ("Oh", "O", "Td", "T", "D4h", "D4", "C4v", "D2d",
                 "D3d", "D3", "C3v", "D2h", "D2", "C2v")

#: Butler (1981)'s branching tables, as the Oh irrep each crystal-field
#: parameter descends through: ``{group: {rank: {Oh irrep: n routes}}}``.
#:
#: **Provenance.** Derived by ``scripts/derive_butler_routes.py`` from
#: ``ttmult/src/disk0``, 271 KB of ASCII headed
#: ``GROUPDATA D4 CREATED FOR_BUTLER(1981) 25_MAR_1980`` -- the same tables
#: ``ttrac`` reads (``ttrac.c:4148-4169`` maps each crystallographic group onto
#: Butler's internal one). The blocks used are ``BRANCHDATA SO3=>O`` and then
#: ``O=>D4`` (tetragonal), ``O=>D3`` (trigonal), ``O=>T`` (T), and
#: ``O=>D4`` + ``D4=>D2`` for the orthorhombic groups, since Butler tabulates
#: no ``O=>D2``. A route's intermediate Butler O label maps to Mulliken by
#: dimension and by the ``SO3=>O`` branching: ``0`` = A1 (rank 0 branches to
#: it), ``^0`` = A2, ``2`` = E (dim 2), ``1`` = T1 (rank 1 branches to it),
#: ``^1`` = T2 (rank 2 branches to E + T2). The crystal field is gerade, so the
#: gerade labels are the ones that can appear.
#:
#: ``Th`` is in the script's chain table but is not a WP-B target, so it is not
#: listed here.
#:
#: :func:`test_butlers_tables_still_say_what_this_transcription_says`
#: re-derives this from ``disk0`` when that file is reachable.
BUTLER_OH_ROUTES = {
    "Oh":  {4: {"A1g": 1}, 2: {}},
    "O":   {4: {"A1g": 1}, 2: {}},
    "Td":  {4: {"A1g": 1}, 2: {}},
    "T":   {4: {"A1g": 1}, 2: {}},
    "D4h": {4: {"A1g": 1, "Eg": 1}, 2: {"Eg": 1}},
    "D4":  {4: {"A1g": 1, "Eg": 1}, 2: {"Eg": 1}},
    "C4v": {4: {"A1g": 1, "Eg": 1}, 2: {"Eg": 1}},
    "D2d": {4: {"A1g": 1, "Eg": 1}, 2: {"Eg": 1}},
    "D3d": {4: {"A1g": 1, "T2g": 1}, 2: {"T2g": 1}},
    "D3":  {4: {"A1g": 1, "T2g": 1}, 2: {"T2g": 1}},
    "C3v": {4: {"A1g": 1, "T2g": 1}, 2: {"T2g": 1}},
    "D2h": {4: {"A1g": 1, "Eg": 2}, 2: {"Eg": 2}},
    "D2":  {4: {"A1g": 1, "Eg": 2}, 2: {"Eg": 2}},
    "C2v": {4: {"A1g": 1, "Eg": 2}, 2: {"Eg": 2}},
}

#: Rank at which the cubic invariant lives. There is no rank-2 Oh invariant,
#: which is why Oh has no Ds.
CUBIC_RANK = 4

#: Singular values of ``P_Gamma V`` above this count as a route. The projector
#: is a sum of 48 Wigner matrices, so the gap is between ~1e-16 and ~0.6.
ROUTE_TOL = 1e-8


def _oh_isotypic_projector(irrep: str, k: int) -> np.ndarray:
    """``P_Gamma`` on ``D^(k)`` in the ``|k, q>`` basis, for an Oh irrep.

    ``P = (d/|Oh|) sum_g conj(chi(g)) D^(k)(g)``. The crystal field is gerade,
    so ``D^(k)`` is the same matrix for an element and its product with
    inversion; the ungerade projectors come out zero, as they must.
    """
    oh = point_group("Oh")
    row = oh.irrep_labels.index(irrep)
    chi, cls = oh.character_table, oh.class_of
    P = np.zeros((2 * k + 1, 2 * k + 1), dtype=complex)
    for j, e in enumerate(oh.elements):
        P += np.conj(chi[row, cls[j]]) * wigner_from_quaternion(float(k), e.quat)
    return P * (oh.irrep_dims[row] / oh.order)


def oh_route_multiplicities(group: str, k: int) -> dict:
    """``{Oh irrep: n}`` for the rank-k crystal-field parameters of *group*.

    The Oh isotypic decomposition of the group's totally symmetric subspace of
    ``D^(k)``. Only irreps with at least one route appear, so the result is
    directly comparable with a Butler route count.
    """
    V = np.asarray(cf_invariant_vectors(group, k))
    if V.shape[1] == 0:
        return {}
    oh = point_group("Oh")
    out = {}
    for irrep in oh.irrep_labels:
        n = int(np.sum(np.linalg.svd(_oh_isotypic_projector(irrep, k) @ V,
                                     compute_uv=False) > ROUTE_TOL))
        if n:
            out[irrep] = n
    return out


def route_mismatch(group: str, expected: dict) -> str:
    """``""`` when *group*'s routes are *expected*, else what differs.

    Both the parametrized test and the injection test below go through this,
    so the injections exercise the same comparison rather than a restatement
    of it (the previous handoff's lesson 1).
    """
    problems = []
    for k in CF_RANKS:
        got = oh_route_multiplicities(group, k)
        if got != expected.get(k, {}):
            problems.append(f"rank {k}: derived {got}, Butler {expected.get(k, {})}")
    return "; ".join(problems)


@pytest.mark.parametrize("group", TARGET_GROUPS)
def test_the_oh_route_multiplicities_match_butlers_branching_tables(group):
    """Every crystal-field parameter descends through the Oh irrep Butler says.

    This is residual 50's 15-of-15 agreement sharpened from *rank-resolved* to
    *route-resolved*, and it is the first independent check on the **trigonal**
    routes: ``derive_butler_routes.py``'s own gate validates its parser on the
    tetragonal chain only. D3d/D3/C3v coming out as Oh A1 + Oh T2 at rank 4 and
    Oh T2 at rank 2 -- T2 exactly where the tetragonal groups have E -- is the
    statement that was previously taken on the parser's word.
    """
    assert route_mismatch(group, BUTLER_OH_ROUTES[group]) == ""


def test_the_route_oracle_rejects_a_wrong_setting_and_a_wrong_family():
    """The injections the route comparison exists to catch, seen to fail.

    Residual 40's bug is the realistic one: the trigonal groups registered in
    the crystallographic z setting instead of the octahedral one. The second is
    the mistake the trigonal routes were never checked against -- a tetragonal
    invariant space where a trigonal one belongs, which has the right parameter
    *count* and the wrong routes, so nothing rank-resolved would see it.
    """
    assert route_mismatch("D3d[z]", BUTLER_OH_ROUTES["D3d"]) != ""
    assert route_mismatch("D4h", BUTLER_OH_ROUTES["D3d"]) != ""
    assert route_mismatch("D3d", BUTLER_OH_ROUTES["D4h"]) != ""
    # ... and a route table that is right in total but wrong in attribution.
    assert route_mismatch("D3d", {4: {"A1g": 1, "Eg": 1}, 2: {"Eg": 1}}) != ""


@pytest.mark.parametrize("group", ["D3d[z]", "D3[z]", "C3v[z]"])
def test_the_z_setting_trigonal_groups_break_the_route_decomposition(group):
    """The decomposition needs ``G <= Oh``, and here is what happens without it.

    ``P_Gamma`` commutes with ``D^(k)(h)`` only for ``h`` in Oh. The z-setting
    trigonal groups are *not* Oh subgroups (8 elements outside for D3d[z];
    ``test_crystal_field.py`` keeps that measurement), so the Oh isotypic
    pieces of their invariant space are not subspaces of it and the
    multiplicities overcount: 4 routes for a 2-parameter rank-4 space. Keeping
    this as a test means the commuting argument above is load-bearing rather
    than decorative.
    """
    n_params = np.asarray(cf_invariant_vectors(group, CUBIC_RANK)).shape[1]
    assert sum(oh_route_multiplicities(group, CUBIC_RANK).values()) > n_params


@pytest.mark.parametrize("group", TARGET_GROUPS)
def test_exactly_one_parameter_of_every_group_is_the_oh_invariant_one(group):
    """"10Dq plus distortions" is exact: one Oh-invariant route and no other.

    One rank-4 route through Oh A1 and none at rank 2, for all 14. The second
    half is what makes the descent limit non-vacuous -- if a distortion were
    secretly Oh-invariant, setting it to zero would be a choice of basis rather
    than a limit.
    """
    routes = {k: oh_route_multiplicities(group, k) for k in CF_RANKS}
    assert routes[CUBIC_RANK].get("A1g") == 1
    assert "A1g" not in routes[2]
    # and the routes account for every parameter, so there is no third kind.
    for k in CF_RANKS:
        assert sum(routes[k].values()) == np.asarray(cf_invariant_vectors(group, k)).shape[1]


def test_ohs_rank_four_invariant_space_is_the_cubic_field_and_nothing_else():
    """The premise the route decomposition rests on, asserted as the premise.

    ``P_A1g`` on ``D^(4)`` is the rank-one projector onto Oh's cubic invariant,
    and there is no Oh invariant at rank 2 at all. So "the A1g route" and "the
    cubic field" are the same thing *by construction*, which is why there is no
    per-group test that the A1g route vector equals the cubic invariant: the
    image of ``P_A1g`` is that line whatever subgroup is on the other side.
    I wrote such a test; injecting the z-setting invariant space for D3d's left
    it passing, which is how it was found to prove nothing. The per-group
    content is the **multiplicity** (route test above) and the **containment**
    (``test_crystal_field.py``'s
    ``test_the_cubic_invariant_lies_inside_every_octahedral_subgroups_invariants``,
    which the z setting fails at residual 0.6455).
    """
    cubic = np.asarray(cf_invariant_vectors("Oh", CUBIC_RANK))[:, 0]
    P = _oh_isotypic_projector("A1g", CUBIC_RANK)
    np.testing.assert_allclose(P, np.outer(cubic, cubic.conj()), atol=1e-12)
    assert np.linalg.norm(_oh_isotypic_projector("A1g", 2)) < 1e-12


@pytest.mark.parametrize("group", TARGET_GROUPS)
def test_ten_dq_is_literally_the_same_operator_vector_in_every_group(group):
    """**The descent limit.** Measured at <= 1.8e-15 on a vector of norm 3.29.

    Oh's own one-electron pattern, solved inside *each* group's invariant
    subspace, returns Oh's own operator vector. So at zero distortion a lower
    group's crystal-field operator is not approximately Oh's, it is the same
    array -- and ``_build_coupling_operator`` is linear in it, so every
    downstream quantity (the RAC ADD coefficients, the many-body Hamiltonian,
    the stick spectrum) is bit-for-bit the Oh one. That is how Oh's Fortran
    validation reaches a group with no Fortran reference, and it is why this is
    asserted on the vector rather than by re-running the pipeline: with
    identical vectors a pipeline comparison would have one thing on both sides.

    The solve is not a restatement of the containment test. It is a
    least-squares inside an ``n``-dimensional space (2 for the uniaxial groups,
    3 for the orthorhombic ones) whose answer would move if the invariant
    space, the subduction gauge or the unit-tensor normalisation were wrong --
    ``test_crystal_field.py`` injects each of those three and watches this kind
    of solve fail. Measured here: serving D3d the **z-setting** invariant space
    makes the solve refuse the pattern outright ("the convention and the
    symmetry disagree").

    **What it cannot catch, measured rather than assumed:** serving D3d
    **D4h's** invariant space leaves it passing, because D4h's rank-4 space
    contains the cubic invariant too. The Oh limit is family-blind by
    construction -- distinguishing one family's distortion routes from
    another's is what
    :func:`test_the_oh_route_multiplicities_match_butlers_branching_tables` is
    for, and it rejects that substitution.
    """
    oh_vector = cf_operator_vector("Oh", "10DQ")[1]
    pattern = ONE_ELECTRON_PATTERNS["Oh"]["10DQ"][1]
    got = cf_operator_from_pattern(group, CUBIC_RANK, pattern)
    assert np.max(np.abs(got - oh_vector)) < 1e-14


@pytest.mark.parametrize("group", TARGET_GROUPS)
def test_at_zero_distortion_the_one_electron_energies_are_ohs(group):
    """eg at +6Dq, t2g at -4Dq, and no splitting the group would otherwise allow.

    The same statement as the test above, read on the observable rather than on
    the operator vector, and the one that says what "zero distortion" buys: a
    D3d field at zero distortion gives **no trigonal splitting of t2g**, even
    though D3d is free to split it. The matrix is diagonal in
    :data:`D_ORBITAL_ORDER` too, which is the corresponding statement about the
    eigenvectors -- the [111] trigonal field mixes the d orbitals, and at zero
    distortion that mixing is absent rather than small.

    Same blind spot as the test above and for the same reason: measured, D4h's
    invariant space substituted for D3d's passes this too. The route test is
    what sees it.
    """
    pattern = np.asarray(ONE_ELECTRON_PATTERNS["Oh"]["10DQ"][1], dtype=float)
    op = cf_operator_from_pattern(group, CUBIC_RANK, pattern)
    matrix = _one_electron_matrix(CUBIC_RANK, op) / np.sqrt(2 * 2 + 1.0)
    np.testing.assert_allclose(matrix, np.diag(pattern), atol=1e-12)
    energies = dict(zip(D_ORBITAL_ORDER, np.diag(matrix)))
    assert energies["z2"] == pytest.approx(0.6) and energies["x2-y2"] == pytest.approx(0.6)
    for t2g in ("xy", "yz", "xz"):
        assert energies[t2g] == pytest.approx(-0.4)


#: Groups whose crystal field can realise the full splitting its d-orbital
#: branching allows. The cubic family cannot and correctly so -- see
#: :func:`test_the_cubic_family_cannot_split_the_d_shell_below_ohs_pattern`.
SPLITTING_GROUPS = tuple(g for g in TARGET_GROUPS
                         if len(BUTLER_OH_ROUTES[g][CUBIC_RANK]) > 1)


def _degeneracies(matrix: np.ndarray, tol: float = 1e-7) -> list:
    """Sorted eigenvalue multiplicities of a one-electron d matrix."""
    out = []
    for e in np.linalg.eigvalsh(matrix):
        if out and abs(e - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([e, 1])
    return sorted(n for _e, n in out)


def _generic_field(group: str, seed: int) -> np.ndarray:
    """A one-electron d matrix at a random point of *group*'s invariant space."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((5, 5))
    for k in CF_RANKS:
        V = np.asarray(cf_invariant_vectors(group, k))
        for i in range(V.shape[1]):
            matrix = matrix + rng.normal() * _one_electron_matrix(k, V[:, i])
    return matrix


@pytest.mark.parametrize("group", SPLITTING_GROUPS)
@pytest.mark.parametrize("seed", [1, 7, 23])
def test_a_generic_field_splits_the_d_shell_the_way_the_branching_says(group, seed):
    """Away from the Oh limit, the degeneracies are the group's own branching.

    The other end of the descent limit, and an independent check on the
    distortion routes: the branching ``D^2 (gerade) -> G`` is computed from
    characters, the splitting from the invariant operators, and they must agree
    on how the five d orbitals group. D3d gives 1 + 2 + 2 (t2g -> a1g + eg, the
    Oh eg pair staying degenerate), D4h gives 1 + 1 + 1 + 2, the orthorhombic
    groups five singlets. Three seeds, because a single random point could land
    on an accidental degeneracy.

    **Orientation-blind, measured:** the z-setting D3d invariant space passes
    this, because the two settings are the same abstract group with the same
    branching (``test_crystal_field.py::test_the_two_settings_are_the_same_abstract_group``).
    What it does catch is a wrong *family* -- D4h's space substituted for D3d's
    splits 1 + 1 + 1 + 2 and is rejected.
    """
    g = point_group(group)
    expected = sorted(int(g.irrep_dims[g.irrep_labels.index(irrep)])
                      for irrep, copies in g.branching(2, +1).items()
                      for _ in range(copies))
    assert _degeneracies(_generic_field(group, seed)) == expected


@pytest.mark.parametrize("group", ["Oh", "O", "Td", "T"])
@pytest.mark.parametrize("seed", [1, 7, 23])
def test_the_cubic_family_cannot_split_the_d_shell_below_ohs_pattern(group, seed):
    """One route means one parameter means Oh's 2 + 3, everywhere in the family.

    Worth asserting separately rather than folding into the test above, because
    for **T** the branching and the splitting genuinely disagree: the d
    orbitals span ``Aa + Ab + T`` there (the two one-dimensional irreps being a
    complex-conjugate pair), so the branching permits 1 + 1 + 3 -- but T's
    invariant space is one-dimensional and *is* Oh's cubic field, so no T
    crystal field can lift the eg degeneracy. The branching says which irreps
    the orbitals span; the invariant space says which splittings a field can
    produce, and they are not the same question.
    """
    assert _degeneracies(_generic_field(group, seed)) == [2, 3]


DISK0 = Path(__file__).resolve().parents[2].parent / "ttmult" / "src" / "disk0"


@pytest.mark.skipif(not DISK0.exists(), reason=f"Butler tables not present at {DISK0}")
def test_butlers_tables_still_say_what_this_transcription_says():
    """Re-derive :data:`BUTLER_OH_ROUTES` from ``disk0`` when it is reachable.

    A drift guard on the transcription, not the oracle itself: ``disk0`` lives
    in the sibling ``ttmult`` tree, is 271 KB, and is not bundled with this
    package, so the table above has to stand on its own for anyone without the
    Fortran suite. When the file *is* there this re-reads it through
    ``scripts/derive_butler_routes.py`` -- whose own gate against
    ``pyctm/write_RAC.py`` runs first and gates the rest.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        import derive_butler_routes as butler
    finally:
        sys.path.pop(0)

    tables = butler.ButlerTables(DISK0)
    assert butler.run_gate(tables, verbose=False), "the script's own gate failed"

    butler_o_to_mulliken = {"0": "A1g", "^0": "A2g", "2": "Eg", "1": "T1g", "^1": "T2g"}
    for group, expected in BUTLER_OH_ROUTES.items():
        chain = butler.CHAIN[group.lower()]
        for k in CF_RANKS:
            counts: dict = {}
            for path in butler.invariant_routes(tables, chain, k):
                label = butler_o_to_mulliken[path[1]]
                counts[label] = counts.get(label, 0) + 1
            assert counts == expected[k], (group, k)
