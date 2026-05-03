"""Status sentinel for the D4h from-scratch port (plan Phase 1).

Each test in this file documents one aspect of the port. They are
xfail-marked until the corresponding piece of the port lands; once it
does, drop the xfail and the test becomes the contract for that piece.

See `~/.claude/plans/lexical-toasting-kahan.md` for the phased plan.
"""
from __future__ import annotations

import math

import pytest


def test_generate_ledge_rac_accepts_sym_oh_default():
    """generate_ledge_rac should accept the sym='oh' kwarg as a no-op default."""
    from multitorch.angular.rac_generator import generate_ledge_rac

    rac, cowan = generate_ledge_rac(
        l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='oh',
    )
    assert rac is not None
    assert isinstance(cowan, list) and len(cowan) == 4


def test_generate_ledge_rac_d4h_raises_with_pointer():
    """sym='d4h' raises NotImplementedError pointing at the Phase 1c work."""
    from multitorch.angular.rac_generator import generate_ledge_rac
    with pytest.raises(NotImplementedError, match="Phase 1c"):
        generate_ledge_rac(
            l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='d4h',
        )


def test_generate_ledge_rac_unknown_sym_raises():
    """Unknown sym raises ValueError with a helpful message."""
    from multitorch.angular.rac_generator import generate_ledge_rac
    with pytest.raises(ValueError, match="Unsupported symmetry"):
        generate_ledge_rac(
            l_val=2, n_val_gs=8, l_core=1, n_core_gs=6, sym='nope',
        )


@pytest.mark.xfail(reason="Phase 1: calcXAS_from_scratch sym kwarg not yet plumbed")
def test_calcXAS_from_scratch_accepts_sym_d4h():
    """calcXAS_from_scratch should dispatch to the D4h path when sym='d4h'."""
    from multitorch.api.calc import calcXAS_from_scratch

    x, y = calcXAS_from_scratch(
        "Ni", "ii", cf={"tendq": 1.0, "ds": 0.0, "dt": 0.0}, sym="d4h",
    )
    assert x.shape == y.shape
    assert y.max() > 0


def test_d4h_butler_coefficients_match_pyctm():
    """D4H_BRANCHES_BY_OPERATOR must match the per-parameter coefficients in
    pyctm/pyctm/write_RAC.py:gen_X400_X420_X220 (lines 349–351):

        X400 = (6/10) * sqrt(30) * tendq  −  (7/2) * sqrt(30) * dt
        X420 = -(5/2) * sqrt(42) * dt
        X220 = -sqrt(70) * ds

    Per-parameter Butler coefficients for unit input:
      tendq=1 → X400_tendq_A1g = +6*sqrt(30)/10
      dt=1    → X400_dt_A1g    = -7*sqrt(30)/2
      dt=1    → X420_dt_Eg     = -5*sqrt(42)/2
      ds=1    → X220_ds_A1g    = -sqrt(70)
    """
    from multitorch.angular.symmetry import D4H_BRANCHES_BY_OPERATOR

    tendq = D4H_BRANCHES_BY_OPERATOR['TENDQ']
    dt_ = D4H_BRANCHES_BY_OPERATOR['DT']
    ds_ = D4H_BRANCHES_BY_OPERATOR['DS']

    # All keys have third slot '0+' (D4h A1g target — CF operators are scalar).
    assert tendq[('4+', '0+', '0+')] == pytest.approx(
        6.0 * math.sqrt(30.0) / 10.0, rel=1e-6,
    )
    assert dt_[('4+', '0+', '0+')] == pytest.approx(
        -7.0 / 2.0 * math.sqrt(30.0), rel=1e-6,
    )
    assert dt_[('4+', '2+', '0+')] == pytest.approx(
        -5.0 / 2.0 * math.sqrt(42.0), rel=1e-6,
    )
    assert ds_[('2+', '2+', '0+')] == pytest.approx(
        -math.sqrt(70.0), rel=1e-6,
    )


def test_d4h_branching_J0_J2_J4():
    """d4h_branching: integer J should compose oh_branching with OH_TO_D4H.

    Sanity checks against well-known L→D4h reductions:
      J=0 (s):  A1g
      J=2 (d):  A1g + B1g + B2g + Eg   (Oh: Eg + T2g; Eg→A1g+B1g; T2g→B2g+Eg)
      J=4 (g):  many irreps, but A1g multiplicity must be ≥ 1
    """
    from multitorch.angular.symmetry import d4h_branching

    j0 = d4h_branching(0)
    assert j0.get('A1g', 0) >= 1
    assert sum(j0.values()) == 1, f"D^0 has dim 1, got branches summing to {sum(j0.values())}"

    j2 = d4h_branching(2)
    # D^2 has dim 5 in D4h: A1g + B1g + B2g + Eg(dim 2) = 1+1+1+2 = 5
    # But d4h_branching returns multiplicities, not dimensions, so we need
    # to weight by D4h irrep dim.
    d4h_dim = {'A1g': 1, 'A2g': 1, 'B1g': 1, 'B2g': 1, 'Eg': 2,
               'A1u': 1, 'A2u': 1, 'B1u': 1, 'B2u': 1, 'Eu': 2}
    total_dim = sum(mult * d4h_dim[irr] for irr, mult in j2.items())
    assert total_dim == 5, f"D^2 should subduce to dim 5, got {total_dim}: {j2}"

    j4 = d4h_branching(4)
    total_dim_j4 = sum(mult * d4h_dim[irr] for irr, mult in j4.items())
    assert total_dim_j4 == 9, f"D^4 should subduce to dim 9, got {total_dim_j4}: {j4}"


def test_d4h_branching_parity_for_odd_L():
    """d4h_branching for odd L should produce ungerade (u) irreps."""
    from multitorch.angular.symmetry import d4h_branching

    j1 = d4h_branching(1)
    # J=1 is T1u in Oh → A2u + Eu in D4h. All irreps must be ungerade.
    for irr in j1:
        assert irr.endswith('u'), f"J=1 should produce ungerade irreps only, got {irr}"


def test_d4h_cf_operator_recipe_tendq():
    """TENDQ recipe: single rank-4 entry via Oh A1, scalar in D4h A1g."""
    from multitorch.angular.symmetry import d4h_cf_operator_recipe
    recipe = d4h_cf_operator_recipe('TENDQ')
    assert len(recipe) == 1
    rank, oh, d4h, coeff = recipe[0]
    assert rank == 4
    assert oh == 'A1'
    assert d4h == 'A1g'
    assert coeff == pytest.approx(6.0 * math.sqrt(30.0) / 10.0, rel=1e-12)


def test_d4h_cf_operator_recipe_dt():
    """DT recipe: two rank-4 entries (Oh A1 and Oh E paths), both → D4h A1g."""
    from multitorch.angular.symmetry import d4h_cf_operator_recipe
    recipe = d4h_cf_operator_recipe('DT')
    assert len(recipe) == 2
    paths = {(rank, oh, d4h): coeff for rank, oh, d4h, coeff in recipe}
    assert paths[(4, 'A1', 'A1g')] == pytest.approx(-7.0 / 2.0 * math.sqrt(30.0), rel=1e-12)
    assert paths[(4, 'E', 'A1g')] == pytest.approx(-5.0 / 2.0 * math.sqrt(42.0), rel=1e-12)


def test_d4h_cf_operator_recipe_ds():
    """DS recipe: single rank-2 entry via Oh E path, → D4h A1g."""
    from multitorch.angular.symmetry import d4h_cf_operator_recipe
    recipe = d4h_cf_operator_recipe('DS')
    assert len(recipe) == 1
    rank, oh, d4h, coeff = recipe[0]
    assert rank == 2
    assert oh == 'E'
    assert d4h == 'A1g'
    assert coeff == pytest.approx(-math.sqrt(70.0), rel=1e-12)


def test_d4h_cf_operator_recipe_unknown_raises():
    from multitorch.angular.symmetry import d4h_cf_operator_recipe
    with pytest.raises(ValueError, match="Unknown D4h CF operator"):
        d4h_cf_operator_recipe('NOPE')


def test_d4h_branching_half_integer_raises():
    """d4h_branching should explicitly fail for half-integer J (Phase 1c)."""
    from multitorch.angular.symmetry import d4h_branching
    with pytest.raises(NotImplementedError, match="half-integer"):
        d4h_branching(0.5)


@pytest.mark.xfail(reason="Phase 1: D4h Ni parity test pending")
def test_d4h_ni_from_scratch_matches_nid8_fixture():
    """D4h Ni from-scratch should match the bundled nid8 fixture within tolerances."""
    from multitorch.api.calc import calcXAS_cached, calcXAS_from_scratch, preload_fixture
    cf = {"tendq": 1.0, "ds": 0.0, "dt": 0.0}
    cache = preload_fixture("Ni", "ii", "d4h")
    _, y_ref = calcXAS_cached(cache, cf=cf)
    _, y_new = calcXAS_from_scratch("Ni", "ii", cf=cf, sym="d4h")
    # Tolerance check would use bench.parity.compare(...).passes(...)
    assert y_new.shape == y_ref.shape
