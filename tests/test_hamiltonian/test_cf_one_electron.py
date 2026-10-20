"""One-electron crystal-field oracle for the D4h/Oh parameter conventions.

With every Slater integral and spin-orbit constant set to zero, a d^8
ground manifold is two holes in a one-electron crystal field, so every
eigenvalue must be a pair sum of one-electron hole energies. That is an
analytic oracle for the *convention* of 10Dq / Dt / Ds independent of
the Fortran fixtures (scientific audit 2026-09, findings S2 and S4).

Ballhausen / CTM4XAS one-electron d energies in D4h (electron picture):

    E(z2)      = 6Dq - 2Ds - 6Dt
    E(x2-y2)   = 6Dq + 2Ds -  Dt
    E(xy)      = -4Dq + 2Ds -  Dt
    E(xz, yz)  = -4Dq -  Ds + 4Dt

Hole energies are the negatives; the d^8 levels are all pair sums
(including the doubly-occupied-hole pairs) of the five hole energies.

**d^1 and d^9 extend the same oracle to half-integer J (added 2026-09-25).**
One electron (d^1) or one hole (d^9) in the field gives the five one-particle
energies directly, with no pair sums, and both have half-integer J — so this is
an analytic, Fortran-free oracle for the double-group D4h emitters
(`_make_d4h_op_adds_half`). It is the only oracle that reaches the **rank-2 (DS)**
coefficients at half-integer J: the Oh-limit parity tests run at dt = ds = 0
where only the Oh-scalar TENDQ acts, so all ~500 DS entries of an odd-electron
D4h RAC were previously unchecked by anything (a scientific audit on 2026-09-25
showed a DS sign flip passing every other half-integer D4h test, including the
dipole sum rule, which is crystal-field-blind by construction).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from multitorch.api.calc import _build_ban_from_rac, _run_phase5_pipeline


def _one_electron_energies(tendq: float, dt: float = 0.0, ds: float = 0.0) -> np.ndarray:
    """The five Ballhausen one-electron d energies in D4h (electron picture)."""
    dq = tendq / 10.0
    return np.array([
        6 * dq - 2 * ds - 6 * dt,   # z2
        6 * dq + 2 * ds - dt,       # x2-y2
        -4 * dq + 2 * ds - dt,      # xy
        -4 * dq - ds + 4 * dt,      # xz
        -4 * dq - ds + 4 * dt,      # yz
    ])


def ballhausen_d8_levels(tendq: float, dt: float = 0.0, ds: float = 0.0) -> np.ndarray:
    """Sorted unique two-hole levels (relative to the lowest) for d^8."""
    holes = -_one_electron_energies(tendq, dt, ds)
    pairs = [holes[i] + holes[j] for i, j in itertools.combinations_with_replacement(range(5), 2)]
    lv = np.unique(np.round(np.array(pairs), 9))
    return lv - lv.min()


def ballhausen_one_particle_levels(tendq: float, dt: float = 0.0, ds: float = 0.0,
                                   hole: bool = False) -> np.ndarray:
    """Sorted unique one-particle levels (relative to the lowest).

    ``hole=False`` is d^1 (one electron, energies as given); ``hole=True`` is d^9
    (one hole, energies negated). Both have half-integer J.
    """
    e = _one_electron_energies(tendq, dt, ds)
    lv = np.unique(np.round(-e if hole else e, 9))
    return lv - lv.min()


def _relative_levels(eigs: torch.Tensor, window: float = 30.0) -> np.ndarray:
    """Unique ground-manifold eigenvalues relative to the lowest one."""
    e = eigs.detach().cpu().numpy()
    e = e[e < e.min() + window]
    lv = np.unique(np.round(e - e.min(), 5))
    return lv


def _fixture_ground_levels(sym: str, cf: dict) -> np.ndarray:
    """Union of ground-state eigenvalues over all triads, Slater = SOC = 0."""
    result = _run_phase5_pipeline(
        "Ni", "ii", sym, "l", cf,
        slater=0.0, soc=0.0,
        delta=100.0, lmct=0.0, mlct=None,
    )
    eigs = torch.cat([t.Eg for t in result.triads])
    return _relative_levels(eigs)


def _from_scratch_ground_levels(sym: str, cf: dict, n_val_gs: int = 8) -> np.ndarray:
    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory

    rac, cowan = generate_ledge_rac(
        l_val=2, n_val_gs=n_val_gs,
        raw_slater_gs_ry={"F0": 0.0, "F2": 0.0, "F4": 0.0},
        raw_zeta_gs_ry=0.0,
        raw_slater_ex_ry={"F2_dd": 0.0, "F4_dd": 0.0, "G1_pd": 0.0,
                          "G3_pd": 0.0, "F2_pd": 0.0},
        raw_zeta_ex_ry={"d": 0.0, "p": 0.0},
        sym=sym,
    )
    # the caller's cf, passed through rather than expanded into every key:
    # expanding it manufactured dt=ds=0.0 for Oh, which WP-B5-4 refuses
    # (residual 43). Every case here names tendq, so no default is in play.
    ban = _build_ban_from_rac(rac, sym=sym, cf=cf)
    result = assemble_and_diagonalize_in_memory(cowan, rac, ban)
    eigs = torch.cat([t.Eg for t in result.triads])
    return _relative_levels(eigs)


# Every key is given explicitly: the nid8ct template carries nonzero Dt/Ds
# and ``modify_ban_params`` only overrides the keys it is handed.
CASES = [
    pytest.param({"tendq": 1.0, "dt": 0.0, "ds": 0.0}, id="10Dq"),
    pytest.param({"tendq": 0.0, "dt": 0.0, "ds": 0.1}, id="Ds"),
    pytest.param({"tendq": 0.0, "dt": 0.1, "ds": 0.0}, id="Dt"),
    pytest.param({"tendq": 1.0, "dt": 0.07, "ds": -0.03}, id="mixed"),
]


@pytest.mark.parametrize("cf", CASES)
def test_fixture_path_d4h_matches_ballhausen(cf):
    """Fixture path (nid8ct template): every CF parameter is Ballhausen."""
    expected = ballhausen_d8_levels(cf.get("tendq", 0.0), cf.get("dt", 0.0), cf.get("ds", 0.0))
    got = _fixture_ground_levels("d4h", cf)
    np.testing.assert_allclose(got, expected, atol=2e-5, err_msg=f"cf={cf}")


def test_fixture_path_oh_matches_ballhausen():
    expected = ballhausen_d8_levels(1.0)
    got = _fixture_ground_levels("oh", {"tendq": 1.0})
    np.testing.assert_allclose(got, expected, atol=2e-5)


def test_from_scratch_oh_matches_ballhausen():
    expected = ballhausen_d8_levels(1.0)
    got = _from_scratch_ground_levels("oh", {"tendq": 1.0})
    np.testing.assert_allclose(got, expected, atol=2e-5)


@pytest.mark.parametrize("cf", CASES + [pytest.param({"tendq": 0.6, "dt": -0.05, "ds": 0.12}, id="mixed2")])
def test_from_scratch_d4h_matches_ballhausen(cf):
    """From-scratch D4h dispatcher (audit S4 closed 2026-09-10: partner-resolved
    Oh→D4h subduction, Euler-angle fix for the C2' rotations, i^(Jk-Jb) gauge
    sign on cross-J crystal-field couplings, Ballhausen sign pin on the
    operator vectors)."""
    expected = ballhausen_d8_levels(cf.get("tendq", 0.0), cf.get("dt", 0.0), cf.get("ds", 0.0))
    got = _from_scratch_ground_levels("d4h", cf)
    np.testing.assert_allclose(got, expected, atol=2e-5, err_msg=f"cf={cf}")


# ─────────────────────────────────────────────────────────────
# Half-integer J: one electron (d1) and one hole (d9)
# ─────────────────────────────────────────────────────────────

HALF_CASES = CASES + [pytest.param({"tendq": 0.6, "dt": -0.05, "ds": 0.12}, id="mixed2")]


@pytest.mark.parametrize("n_val_gs,hole", [(1, False), (9, True)], ids=["d1", "d9"])
@pytest.mark.parametrize("cf", HALF_CASES)
def test_from_scratch_d4h_half_integer_matches_ballhausen(cf, n_val_gs, hole):
    """The double-group D4h emitters against the analytic one-particle energies.

    This is the oracle for the rank-2 (DS) coefficients at half-integer J — see
    the module docstring. The ``Ds`` and ``mixed``/``mixed2`` cases are the ones
    that carry it; ``10Dq`` and ``Dt`` also pin the rank-4 pair, independently of
    the Oh-limit fixtures.
    """
    expected = ballhausen_one_particle_levels(
        cf.get("tendq", 0.0), cf.get("dt", 0.0), cf.get("ds", 0.0), hole=hole)
    got = _from_scratch_ground_levels("d4h", cf, n_val_gs=n_val_gs)
    np.testing.assert_allclose(got, expected, atol=2e-5, err_msg=f"cf={cf} n={n_val_gs}")


@pytest.mark.parametrize("n_val_gs,hole", [(1, False), (9, True)], ids=["d1", "d9"])
def test_from_scratch_oh_half_integer_matches_ballhausen(n_val_gs, hole):
    expected = ballhausen_one_particle_levels(1.0, hole=hole)
    got = _from_scratch_ground_levels("oh", {"tendq": 1.0}, n_val_gs=n_val_gs)
    np.testing.assert_allclose(got, expected, atol=2e-5)


# ─────────────────────────────────────────────────────────────
# D3d: the trigonal convention (residual 53)
# ─────────────────────────────────────────────────────────────
#
# The trigonal counterpart of this module's D4h table. It is the oracle that
# WP-B6 was narrowed down to -- residual 50 said the Ballhausen Dsigma/Dtau
# assignment was the one genuine lookup left, and residual 52 that it was the
# *only* one, the BRANCH strengths having turned out to have no non-test caller.
#
# PROVENANCE. S. Kaizaki, "Ligand Field Theory", chapter 1 of *Instrumental
# Analysis of Coordination Compounds*, volume 1, eds. H. Oshio and
# G. N. Newton, Coordination Chemistry Fundamentals Series vol. 4, Royal
# Society of Chemistry, 2024, pp. 1-26, doi 10.1039/9781837674978-00001;
# section 1.2.5 "Ligand Field Parameters for Low Symmetry Field", unnumbered
# display equations in the body text (not a table), with Figure 1.13 as the
# splitting diagram. Tier T2 in ``NUMBER_AUDIT.md`` -- exact rational
# coefficients quoted from body text, no uncertainty attached and none needed.
# Read through the publisher's HTML rendering, not the printed chapter; the
# caveats are in the ledger.
#
# These are *diagonal* energies in the trigonal-adapted basis. The chapter
# gives no off-diagonal term between the two e sets, and the general trigonal
# field has one; that it is nevertheless determined is the content of
# ``test_the_published_diagonal_table_determines_both_operators``.

#: ``{level: (coefficient of 10Dq, of Dsigma, of Dtau)}``, transcribed.
KAIZAKI_TRIGONAL = {
    "a1(t2g)": (-4.0, -2.0, -6.0),
    "e(t2g)": (-4.0, 1.0, 2.0 / 3.0),
    "e(eg)": (6.0, 0.0, 7.0 / 3.0),
}

#: Also given in the same paragraph, and an independent fourth coefficient
#: rather than a restatement of the three rows: it is their difference, so a
#: transcription slip in either row would break it.
KAIZAKI_T2G_SPLITTING = (3.0, 20.0 / 3.0)       # (Dsigma, Dtau) of e(t2g) - a1(t2g)

D3D_OPERATORS = ("TENDQ", "DSIGMA", "DTAU")


def _d3d_one_electron_matrices():
    """``{operator: 5x5 one-electron matrix per unit parameter}`` for D3d."""
    from multitorch.angular.crystal_field import _one_electron_matrix, cf_operator_vector

    out = {}
    for operator in D3D_OPERATORS:
        k, vector = cf_operator_vector("D3d", operator)
        out[operator] = _one_electron_matrix(k, vector) / np.sqrt(5.0)
    return out


def _trigonal_basis():
    """``{level: orthonormal columns}`` for a1(t2g), e(t2g) and e(eg).

    Built from the **group** and from Oh's cubic field, never from the two
    operators under test: ``point_group("D3d").subduction(2, +1)`` splits the d
    shell into its A1g column and its two Eg copies by characters alone, and the
    cubic field -- Fortran-validated, and the parent-group label is what "from
    t2g" and "from eg" *mean* -- separates the two Eg copies at -0.4 and +0.6.
    A basis chosen with ``DSIGMA`` would make the test below true by
    construction, which is the previous handoff's lesson 1.
    """
    from multitorch.angular.crystal_field import _one_electron_matrix, cf_operator_vector
    from multitorch.angular.finite_group import point_group
    from multitorch.angular.point_group import _c2r_unitary

    group = point_group("D3d")
    subduction, labels = group.subduction(2, +1)
    basis = _c2r_unitary(2) @ subduction          # into the real-SH order the matrices use
    a1 = basis[:, [i for i, (irrep, _c, _p) in enumerate(labels) if irrep == "A1g"]]
    eg_span = basis[:, [i for i, (irrep, _c, _p) in enumerate(labels) if irrep == "Eg"]]
    assert a1.shape[1] == 1 and eg_span.shape[1] == 4, (a1.shape, eg_span.shape)

    k, vector = cf_operator_vector("Oh", "10DQ")
    cubic = _one_electron_matrix(k, vector) / np.sqrt(5.0)
    levels, mixing = np.linalg.eigh(eg_span.conj().T @ cubic @ eg_span)
    assert np.allclose(np.sort(levels), [-0.4, -0.4, 0.6, 0.6], atol=1e-10), levels
    return {"a1(t2g)": a1,
            "e(t2g)": eg_span @ mixing[:, levels < 0],
            "e(eg)": eg_span @ mixing[:, levels > 0]}


@pytest.mark.parametrize("level", sorted(KAIZAKI_TRIGONAL))
def test_the_trigonal_convention_reproduces_the_published_energies(level):
    """Nine coefficients, no fitted constant, measured to <= 1e-9.

    The convention has **no freedom left** once ``DSIGMA``/``DTAU`` are declared
    to be D4h's ``DS``/``DT`` patterns in the trigonal frame: the rank-2
    invariant space of D3d is one-dimensional and the rank-4 one is spanned by
    the cubic invariant plus the axial operator, so the three rows below are a
    prediction and not a fit. The 1/3 : 2/3 composition of the two e sets, which
    is what turns the axial patterns (-2, -1, +2) and (-6, +4, -1) into
    (-2, +1, 0) and (-6, 2/3, 7/3), is itself derived from the cubic field in
    the trigonal frame.
    """
    matrices = _d3d_one_electron_matrices()
    columns = _trigonal_basis()[level]
    expected = KAIZAKI_TRIGONAL[level]
    scale = {"TENDQ": 10.0, "DSIGMA": 1.0, "DTAU": 1.0}
    for operator, want in zip(D3D_OPERATORS, expected):
        got = np.diag(columns.conj().T @ matrices[operator] @ columns) * scale[operator]
        assert np.max(np.abs(got.imag)) < 1e-10
        assert np.allclose(got.real, want, atol=1e-9), (level, operator, got, want)


def test_the_published_t2g_splitting_relation_holds():
    """``E(e(t2g)) - E(a1(t2g)) = 3 Dsigma + (20/3) Dtau``, the same paragraph."""
    matrices = _d3d_one_electron_matrices()
    basis = _trigonal_basis()

    def energy(level, operator):
        columns = basis[level]
        return float(np.diag(columns.conj().T @ matrices[operator] @ columns).real[0])

    for operator, want in zip(("DSIGMA", "DTAU"), KAIZAKI_T2G_SPLITTING):
        got = energy("e(t2g)", operator) - energy("a1(t2g)", operator)
        assert got == pytest.approx(want, abs=1e-9), operator
    # and the cubic field does not enter the splitting at all, both being t2g
    assert energy("e(t2g)", "TENDQ") - energy("a1(t2g)", "TENDQ") == pytest.approx(0.0, abs=1e-10)


#: Named D3d operators by rank. One per rank-2 parameter and two per rank-4 one,
#: which is ``cf_parameter_count("D3d")`` -- asserted below, so the span used in
#: the injectivity test is the *whole* Hermitian invariant space of that rank and
#: not a slice of it.
D3D_OPERATORS_BY_RANK = {2: ("DSIGMA",), 4: ("TENDQ", "DTAU")}


def _diagonal_columns(operators):
    """``(3, n)`` of each operator's three trigonal diagonal energies."""
    matrices = _d3d_one_electron_matrices()
    basis = _trigonal_basis()
    return np.array([[float(np.diag(basis[level].conj().T @ matrices[op]
                                    @ basis[level]).real[0])
                      for op in operators]
                     for level in ("a1(t2g)", "e(t2g)", "e(eg)")])


@pytest.mark.parametrize("rank", sorted(D3D_OPERATORS_BY_RANK))
def test_the_published_table_determines_each_named_operator(rank):
    """Why the omitted off-diagonal term is forced rather than chosen.

    The chapter lists three diagonal energies and no e(t2g)/e(eg) mixing, while
    the general trigonal field has one. That is not a gap in the convention,
    because *within one rank* the diagonal map is injective: there is no second
    invariant of that rank sharing a published row, so fixing the row fixes the
    operator and with it the mixing. Hermiticity cuts a rank's ``n``-complex-
    dimensional invariant space to ``n`` real dimensions, which is why ``n``
    named operators span all of it.

    Measured rather than argued -- "it must be determined" is the kind of claim
    residual 46 is about, and the first version of this test asserted the wrong
    matrix (see :func:`test_the_diagonal_energies_do_not_determine_the_triple`).
    """
    from multitorch.angular.crystal_field import cf_parameter_count

    operators = D3D_OPERATORS_BY_RANK[rank]
    assert len(operators) == cf_parameter_count("D3d")[rank]
    columns = _diagonal_columns(operators)
    assert np.linalg.matrix_rank(columns, tol=1e-9) == len(operators)
    assert np.linalg.svd(columns, compute_uv=False)[-1] > 1e-3


def test_the_diagonal_energies_do_not_determine_the_triple():
    """But the three energies *together* do not invert, and that is worth pinning.

    The first version of the test above asserted that the 3x3 map from
    ``(10Dq, Dsigma, Dtau)`` to the three diagonal energies is invertible. It is
    not: every crystal field is traceless, so ``E(a1) + 2 E(e(t2g)) +
    2 E(e(eg)) = 0`` and the three rows carry only **two** independent numbers.
    Its rank is 2 and the kernel is ``(10Dq, Dsigma, Dtau) ∝ (-35/9, -20/9, 1)``
    -- a trigonal field of norm ~11 whose three diagonal energies all vanish
    while its e(t2g)/e(eg) mixing has norm 70/9.

    Consequence worth stating where someone fitting will see it: **the one-
    electron level energies alone cannot recover the parameter triple.** The
    mixing is what distinguishes the kernel direction, and it is exactly what
    the published diagonal table omits. The per-rank injectivity above is
    untouched by this -- it is what makes the *convention* well defined -- and
    the two statements are different questions about the same matrix.
    """
    matrices = _d3d_one_electron_matrices()
    basis = _trigonal_basis()
    jacobian = _diagonal_columns(D3D_OPERATORS)
    assert np.linalg.matrix_rank(jacobian, tol=1e-9) == 2

    kernel = np.linalg.svd(jacobian)[2][-1]
    kernel = kernel / kernel[D3D_OPERATORS.index("DTAU")]
    np.testing.assert_allclose(kernel, [-35.0 / 9.0, -20.0 / 9.0, 1.0], atol=1e-9)

    operator = sum(c * matrices[op] for c, op in zip(kernel, D3D_OPERATORS))
    for level in ("a1(t2g)", "e(t2g)", "e(eg)"):
        columns = basis[level]
        assert abs(np.diag(columns.conj().T @ operator @ columns).real[0]) < 1e-9
    mixing = basis["e(t2g)"].conj().T @ operator @ basis["e(eg)"]
    assert np.linalg.norm(mixing) == pytest.approx(70.0 / 9.0, abs=1e-8)
    assert np.linalg.norm(operator) > 1.0, "the kernel direction is a real field, not zero"


def test_the_omitted_eg_mixing_is_proportional_to_dsigma_minus_five_thirds_dtau():
    """The derived consequence, stated so a later change cannot move it quietly.

    The cubic field does not mix the two e sets (0.0 exactly -- they are its own
    eigenspaces). ``DSIGMA`` and ``DTAU`` each do, with Frobenius norms 2 and
    10/3, and their blocks are anti-proportional, so the mixing vanishes on the
    line ``Dsigma = (5/3) Dtau`` and nowhere else. Nothing published was
    consulted for this; it follows from the two operators the table fixed.
    """
    matrices = _d3d_one_electron_matrices()
    basis = _trigonal_basis()

    def block(operator):
        return basis["e(t2g)"].conj().T @ matrices[operator] @ basis["e(eg)"]

    assert np.linalg.norm(block("TENDQ")) < 1e-10
    assert np.linalg.norm(block("DSIGMA")) == pytest.approx(2.0, abs=1e-9)
    assert np.linalg.norm(block("DTAU")) == pytest.approx(10.0 / 3.0, abs=1e-9)
    assert np.linalg.norm(block("DSIGMA") * (5.0 / 3.0) + block("DTAU")) < 1e-9
    # off that line it is present, so "no mixing" is a choice of parameters and
    # not a property of the group.
    assert np.linalg.norm(block("DSIGMA") + block("DTAU")) > 1.0


@pytest.mark.parametrize("tetragonal,trigonal,rank",
                         [("DS", "DSIGMA", 2), ("DT", "DTAU", 4)])
def test_dsigma_and_dtau_are_ds_and_dt_rotated_onto_the_trigonal_axis(tetragonal, trigonal, rank):
    """The claim behind the shared pattern tuples, as an operator identity.

    ``D^(k)(R) v_D3d = v_D4h`` to <= 3.6e-14 on vectors of norm 8.37 and 25.1,
    where ``R`` carries [111] onto z. So the trigonal pair is not *analogous* to
    the tetragonal pair, it is the same two operators in another frame -- which
    is what lets one published table and one Fortran-validated one check each
    other, and what makes the normalisation (sqrt(70) and -sqrt(630)) inherited
    rather than fitted.
    """
    from multitorch.angular.crystal_field import cf_operator_vector
    from multitorch.angular.finite_group import (point_group, quaternion_carrying_to_z,
                                                 wigner_from_quaternion)

    rotation = wigner_from_quaternion(
        float(rank), quaternion_carrying_to_z(point_group("D3d").principal_axis))
    tetragonal_vector = cf_operator_vector("D4h", tetragonal)[1]
    trigonal_vector = cf_operator_vector("D3d", trigonal)[1]
    assert np.max(np.abs(rotation @ trigonal_vector - tetragonal_vector)) < 1e-13


def test_swapping_the_trigonal_patterns_breaks_the_published_table(monkeypatch):
    """Inject the mistake this oracle exists to catch: Dsigma and Dtau exchanged.

    The ranks differ, so a straight swap would not even solve; the injection
    that *does* solve is to give ``DTAU`` the rank-4 **cubic** pattern, which is
    a genuine rank-4 invariant of D3d and is what "the wrong one of the two
    rank-4 routes" means. The published table must reject it.
    """
    from multitorch.angular import crystal_field as cf

    patched = dict(cf.ONE_ELECTRON_PATTERNS)
    patched["D3d"] = dict(cf.ONE_ELECTRON_PATTERNS["D3d"])
    patched["D3d"]["DTAU"] = (4, cf.CUBIC_RANK_4, "cubic")
    monkeypatch.setattr(cf, "ONE_ELECTRON_PATTERNS", patched)

    with pytest.raises(AssertionError):
        test_the_trigonal_convention_reproduces_the_published_energies("e(eg)")


def test_declaring_the_axial_patterns_cubic_is_refused(monkeypatch):
    """And the frame is not decoration: the axial pattern is not a cubic one.

    Residual 47's trap in its new place. For D4h the two frames coincide and
    mislabelling costs nothing, which is why this has to be injected on a
    trigonal group: ``DSIGMA``'s pattern read in the registered axes asks for a
    diagonal operator that D3d's symmetry does not have, and the solve says so.
    """
    from multitorch.angular import crystal_field as cf

    with pytest.raises(ValueError, match="not a rank-2 invariant"):
        cf.cf_operator_from_pattern("D3d", 2, cf.AXIAL_RANK_2, frame="cubic")
