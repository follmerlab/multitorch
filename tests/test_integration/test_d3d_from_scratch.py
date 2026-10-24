"""D3d end-to-end — the fourth group reachable from ``calcXAS``, even counts only.

**What is new here and what is not.** The group theory landed earlier: the
single-group Oh->D3d subduction is gated bit-exact against the tetragonal one
(residual 59), and the ``Dsigma``/``Dtau`` convention was pinned against the
published trigonal one-electron energies with zero free parameters (residual
53, checked in ``tests/test_hamiltonian/test_cf_one_electron.py``). This file
is the wiring: a ``sym='d3d'`` RAC, its store, and a spectrum.

**The oracle is the descent limit, and it inherits Oh's Fortran validation.**
At ``dsigma = dtau = 0`` the D3d crystal-field operator is the *same array* as
Oh's cubic one (residual 51, <= 1.8e-15), so a from-scratch D3d spectrum at
zero distortion must reproduce the Oh spectrum -- and Oh is validated against
ttban for every bundled ion. Measured here at **2.3e-14 (Ni) and 1.4e-14
(Fe)**, which is accumulation through the subduction rotation and the
diagonalisation rather than bit-equality: the 10-21 handoff hoped for "bit for
bit" and that is not what it is, so the tolerance is stated rather than
implied. Away from the limit the spectrum moves by ~0.56, which is what gives
the comparison teeth.

**Half-integer J is refused, and the reason is a property of the group.** D3's
double group has two **one-dimensional** spinor irreps forming a
complex-conjugate pair: Frobenius-Schur indicator 0, where every D4h double
irrep is pseudo-real at -1. A complex Hermitian block is reducible to real
symmetric by a diagonal phase transformation iff all its loop products
``H_ij H_jk H_ki`` are real, and loop products are invariant under exactly the
phase freedom gauge-fixing has available. For D3d's ``1E3/2`` that obstruction
measures **9.9e-01**. So there is no gauge, and the RAC store holds real
coefficients -- which is why odd electron counts raise instead of returning a
number. See residual 68; ``test_d3d_double_group_obstruction.py`` measures it.
"""
from __future__ import annotations

import pytest
import torch

from multitorch.api.calc import calcXAS_from_scratch, preload_from_scratch

GRID = dict(nbins=400)
POOL = dict(max_gs=1)

#: Even 3d electron counts, the ones D3d supports. (element, valence, n_d)
EVEN_IONS = [("Ni", "ii", 8), ("Fe", "ii", 6)]


@pytest.mark.parametrize("element,valence,n_d", EVEN_IONS,
                         ids=[f"{e}{v}" for e, v, _n in EVEN_IONS])
def test_the_descent_limit_reproduces_the_oh_spectrum(element, valence, n_d):
    """Dsigma = Dtau = 0 is Oh, and that is how Oh's Fortran parity transfers."""
    _x, y_oh = calcXAS_from_scratch(element, valence, cf={"tendq": 1.0},
                                    sym="oh", **POOL, **GRID)
    _x, y_d3d = calcXAS_from_scratch(element, valence,
                                     cf={"tendq": 1.0, "dsigma": 0.0, "dtau": 0.0},
                                     sym="d3d", **POOL, **GRID)
    worst = float((y_oh - y_d3d).abs().max())
    assert worst < 1e-12, worst
    # ...and the comparison is not vacuous: the scale of the spectrum is O(1)
    assert float(y_oh.abs().max()) > 1e-3


def test_a_trigonal_distortion_moves_the_spectrum():
    """Without this the descent-limit test would pass on a D3d that did nothing."""
    _x, y_oh = calcXAS_from_scratch("Ni", "ii", cf={"tendq": 1.0},
                                    sym="oh", **POOL, **GRID)
    for cf in ({"tendq": 1.0, "dsigma": 0.3}, {"tendq": 1.0, "dtau": 0.2}):
        _x, y = calcXAS_from_scratch("Ni", "ii", cf=cf, sym="d3d", **POOL, **GRID)
        assert float((y_oh - y).abs().max()) > 1e-2, cf


def test_the_rac_carries_d3d_irreps_and_the_trigonal_operators():
    """D3d must be distinguishable from Oh, or ``sym='d3d'`` is an alias.

    D3d's Butler strings coincide with D4h's -- ``0+``/``^0+``/``1+`` for
    A1g/A2g/Eg -- because ``O=>D3`` branches T1u to A2 + E exactly as ``O=>D4``
    does. What distinguishes the groups is the *branching*: Oh's five irreps
    collapse onto three here, and T2g splits as A1g + Eg where D4h gives
    B2g + Eg.
    """
    from multitorch.angular.rac_generator import generate_ledge_template

    rac, _store, _dec = generate_ledge_template(2, 8, sym="d3d")
    geometries = {b.geometry for b in rac.blocks}
    assert {"HAMILTONIAN", "10DQ", "DSIGMA", "DTAU"} <= geometries, sorted(geometries)
    assert "DT" not in geometries and "DS" not in geometries, sorted(geometries)
    # three gerade irreps, not Oh's five and not D4h's five
    gerade = {b.bra_sym for b in rac.blocks if b.bra_sym.endswith("+")}
    assert gerade == {"0+", "^0+", "1+"}, sorted(gerade)
    # the dipole is uniaxial: PERP and PARA, as D4h, not Oh's single MULTIPOLE
    assert {b.geometry for b in rac.blocks if b.kind == "TRANSI"} == {"PERP", "PARA"}


def test_odd_electron_counts_are_refused_with_the_group_theoretic_reason():
    """Not "not implemented yet" -- there is no real-coefficient form (residual 68)."""
    for element, valence in (("Co", "ii"), ("Cu", "ii"), ("Ti", "iii")):
        with pytest.raises(NotImplementedError, match="Frobenius-Schur"):
            preload_from_scratch(element, valence, "d3d")


def test_d3d_refuses_the_crystal_field_keys_it_has_no_operator_for():
    """``dt``/``ds`` are D4h's names; D3d's are ``dsigma``/``dtau`` (residual 43)."""
    from multitorch.api.calc import calcXAS_cached

    cache = preload_from_scratch("Ni", "ii", "d3d")
    for bad in ({"tendq": 1.0, "dt": 0.1}, {"ds": 0.1}):
        with pytest.raises(ValueError, match="crystal-field keys"):
            calcXAS_cached(cache, cf=bad, nbins=200)


def test_charge_transfer_is_refused_for_d3d():
    """CT beyond Oh/D4h is carried-forward work, and the D3d channel names are
    placeholders precisely because this is the only path that would use them."""
    with pytest.raises(ValueError, match="unsupported symmetry"):
        preload_from_scratch("Ni", "ii", "d3d", charge_transfer=True)
