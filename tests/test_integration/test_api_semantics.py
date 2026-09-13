"""WP-S S6: public parameter semantics of the fixture path.

* ``delta``/``u`` follow pyctm ``writeBAN`` (EG2 = Δ, EF2 = Δ − u) all the way
  into the spectrum.
* Sticks without intensity do not set the energy window or the L3/L2 split.
  Oracle: with V = 0 the ligand-hole final states cannot be reached by the
  dipole operator from the d^n ground state, so their intensity is exactly 0.
* ``med_energy`` is honoured as an absolute crossover on the stick scale.
"""
from __future__ import annotations

import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcXAS, _stick_window
from multitorch.spectrum.broaden import pseudo_voigt

NI = dict(element="Ni", valence="ii", sym="oh", edge="l")


def test_fixture_defaults_equal_explicit_pyctm_parameters():
    """ni2_d8_oh is Δ=5, u=6, V=(2, 1), 10Dq=1: passing them changes nothing."""
    x0, y0 = calcXAS(**NI, cf={})
    x1, y1 = calcXAS(**NI, cf={"tendq": 1.0}, delta=5.0, u=6.0, lmct={"eg": 2.0, "t2g": 1.0},
                     xmin=float(x0[0]), xmax=float(x0[-1]))
    assert torch.allclose(y0, y1, atol=1e-12)


def test_u_moves_the_final_state_ct_energy():
    kw = dict(cf={}, delta=5.0, xmin=5.0, xmax=45.0, return_sticks=True)
    _, _, s6 = calcXAS(**NI, u=6.0, **kw)
    _, _, s3 = calcXAS(**NI, u=3.0, **kw)
    assert (s6[:, 0] - s3[:, 0]).abs().max() > 0.1


def test_dark_ligand_hole_states_do_not_set_the_window():
    x, y, st = calcXAS(**NI, cf={}, delta=100.0, lmct=0.0, return_sticks=True)
    E, M = st[:, 0], st[:, 1]
    dark = M <= 1e-10 * M.max()
    assert dark.any() and E[dark].min() > E[~dark].max() + 50.0   # the decoupled CT manifold
    assert float(x[0]) == float(E[~dark].min()) - 5.0
    assert float(x[-1]) == float(E[~dark].max()) + 5.0


def test_med_energy_is_honoured():
    kw = dict(cf={}, xmin=10.0, xmax=45.0, nbins=800, return_sticks=True, gamma1=0.2, gamma2=0.8)
    x, y_auto, st = calcXAS(**NI, **kw)
    _, y_set, _ = calcXAS(**NI, med_energy=30.0, **kw)
    ref = pseudo_voigt(x, st[:, 0], st[:, 1], fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.8,
                       med_energy=30.0, mode="legacy")
    assert torch.allclose(y_set, ref, atol=1e-12)
    # Ni L3 ends near 22 eV and L2 starts near 32 eV: any split in between is the same
    _, _, med_auto = _stick_window(st[:, 0], st[:, 1], None, None, None)
    assert 22.0 < med_auto < 32.0
    assert torch.allclose(y_auto, y_set, atol=1e-12)


def test_stick_window_ignores_zero_intensity():
    E = torch.tensor([10.0, 12.0, 20.0, 90.0], dtype=DTYPE)
    M = torch.tensor([1.0, 0.5, 0.2, 0.0], dtype=DTYPE)
    assert _stick_window(E, M, None, None, None) == (5.0, 25.0, 15.0)
    assert _stick_window(E, M, 0.0, None, 18.0) == (0.0, 25.0, 18.0)
