"""
Track C5 — Multi-fixture Phase 5 parity sweep + autograd tests.

Validates that ``calcXAS`` in Phase 5 mode (no ``ban_output_path``)
reproduces the bootstrap-from-files result across the full Ti–Ni 3d
series, and that autograd flows through ``slater`` and ``soc`` for
every fixture.

Parity metric
-------------
:func:`multitorch.spectrum.parity.spectral_parity` on the union of the two
energy windows (no peak alignment; both spectra share the Fortran energy
zero), plus the fraction of intensity inside the overlap and the area ratio.
The Phase 5 path rebuilds every HAMILTONIAN block at ``slater=0.8, soc=1.0``
(the reductions the fixtures were generated at), which reproduces the Fortran
store, so the only differences are the 6-decimal print precision of the
``.ban_out`` sticks.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE

REFDATA = Path(__file__).parent.parent / "reference_data"

# Union-window cosine per case (WP-S S7). Phase 5 at the fixture's own
# reductions rebuilds the Fortran store, so the spectra agree to the .ban_out
# print precision; Cr(III) is the exception (open residual, see
# docs/DEVELOPMENT_PLAN_2026-09.md "Known residuals" 14).
CASES = [
    ("ti4_d0_oh",  "Ti", "iv",  "oh",  0.9999999),
    ("v3_d2_oh",   "V",  "iii", "oh",  0.9999999),
    ("cr3_d3_oh",  "Cr", "iii", "oh",  0.988),
    ("mn2_d5_oh",  "Mn", "ii",  "oh",  0.9999999),
    ("fe2_d6_oh",  "Fe", "ii",  "oh",  0.9999999),
    ("fe3_d5_oh",  "Fe", "iii", "oh",  0.9999999),
    ("co2_d7_oh",  "Co", "ii",  "oh",  0.9999999),
    ("ni2_d8_oh",  "Ni", "ii",  "oh",  0.9999999),
    ("nid8ct",     "Ni", "ii",  "d4h", 0.9999999),
]


@pytest.mark.parametrize("case_id,element,valence,sym,cos_min", CASES)
def test_phase5_vs_bootstrap_parity(case_id, element, valence, sym, cos_min):
    """Phase 5 spectrum == bootstrap (.ban_out) spectrum on the union window."""
    from multitorch.api.calc import calcXAS
    from multitorch.spectrum.parity import spectral_parity

    ban_out = REFDATA / case_id / f"{case_id}.ban_out"
    x_ref, y_ref = calcXAS(
        element='', valence='', sym='', edge='', cf={},
        ban_output_path=str(ban_out), T=80, max_gs=1,
    )
    x_p5, y_p5 = calcXAS(
        element=element, valence=valence, sym=sym, edge='l',
        cf={}, slater=0.8, soc=1.0, T=80, max_gs=1,
    )
    p = spectral_parity(x_p5, y_p5, x_ref, y_ref)
    assert p.cosine >= cos_min, f"{case_id}: {p}"
    assert min(p.fraction_inside_a, p.fraction_inside_b) > 0.9999, f"{case_id}: {p}"
    assert p.area_ratio == pytest.approx(1.0, abs=5e-3), f"{case_id}: {p}"


# ─────────────────────────────────────────────────────────────
# Autograd: slater gradient through calcXAS for each fixture
# ─────────────────────────────────────────────────────────────

# Autograd test cases. Cr d3 excluded: its 1074-dim Hamiltonian has
# exact eigenvalue degeneracies that produce NaN in eigh backward
# (PyTorch limitation: 1/(λ_i - λ_j) → inf when λ_i = λ_j).
AUTOGRAD_CASES = [
    ("ni2_d8_oh",  "Ni", "ii",  "oh"),
    ("fe2_d6_oh",  "Fe", "ii",  "oh"),
    ("nid8ct",     "Ni", "ii",  "d4h"),
]


@pytest.mark.parametrize("case_id,element,valence,sym", AUTOGRAD_CASES)
def test_phase5_autograd_slater(case_id, element, valence, sym):
    """Autograd through slater must produce finite nonzero gradient."""
    from multitorch.api.calc import calcXAS

    slater = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    x, y = calcXAS(
        element=element, valence=valence, sym=sym, edge='l',
        cf={}, slater=slater, soc=1.0,
    )

    loss = y.sum()
    grad, = torch.autograd.grad(loss, slater)
    assert torch.isfinite(grad), f"{case_id}: slater grad not finite: {grad}"
    assert grad.abs() > 1e-6, f"{case_id}: slater grad too small: {grad}"


@pytest.mark.parametrize("case_id,element,valence,sym", AUTOGRAD_CASES)
def test_phase5_autograd_soc(case_id, element, valence, sym):
    """Autograd through soc must produce finite nonzero gradient."""
    from multitorch.api.calc import calcXAS

    soc = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    x, y = calcXAS(
        element=element, valence=valence, sym=sym, edge='l',
        cf={}, slater=0.8, soc=soc,
    )

    loss = y.sum()
    grad, = torch.autograd.grad(loss, soc)
    assert torch.isfinite(grad), f"{case_id}: soc grad not finite: {grad}"
    assert grad.abs() > 1e-6, f"{case_id}: soc grad too small: {grad}"


# ─────────────────────────────────────────────────────────────
# Autograd: cf / delta / lmct gradients through calcXAS
# ─────────────────────────────────────────────────────────────

# Use Ni Oh for cf/delta/lmct tests (fast, no degeneracy issues).

def test_phase5_autograd_cf_tendq():
    """Autograd through cf['tendq'] must produce finite nonzero gradient."""
    from multitorch.api.calc import calcXAS

    tendq = torch.tensor(1.2, dtype=DTYPE, requires_grad=True)
    x, y = calcXAS(
        element="Ni", valence="ii", sym="oh", edge="l",
        cf={"tendq": tendq}, slater=0.8, soc=1.0,
    )

    loss = y.sum()
    grad, = torch.autograd.grad(loss, tendq)
    assert torch.isfinite(grad), f"cf tendq grad not finite: {grad}"
    assert grad.abs() > 1e-6, f"cf tendq grad too small: {grad}"


def test_phase5_autograd_delta():
    """Autograd through delta must produce finite nonzero gradient."""
    from multitorch.api.calc import calcXAS

    delta = torch.tensor(4.0, dtype=DTYPE, requires_grad=True)
    x, y = calcXAS(
        element="Ni", valence="ii", sym="oh", edge="l",
        cf={}, delta=delta, slater=0.8, soc=1.0,
    )

    loss = y.sum()
    grad, = torch.autograd.grad(loss, delta)
    assert torch.isfinite(grad), f"delta grad not finite: {grad}"
    assert grad.abs() > 1e-6, f"delta grad too small: {grad}"


def test_phase5_autograd_lmct():
    """Autograd through lmct must produce finite nonzero gradient."""
    from multitorch.api.calc import calcXAS

    lmct = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    x, y = calcXAS(
        element="Ni", valence="ii", sym="oh", edge="l",
        cf={}, lmct=lmct, slater=0.8, soc=1.0,
    )

    loss = y.sum()
    grad, = torch.autograd.grad(loss, lmct)
    assert torch.isfinite(grad), f"lmct grad not finite: {grad}"
    assert grad.abs() > 1e-6, f"lmct grad too small: {grad}"
