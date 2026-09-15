"""Charge transfer from scratch against the Fortran chain (WP-C, increment C4).

Oracle: the bundled pyctm/ttmult two-configuration fixtures run through the
fixture path (store rebuilt at its own Slater reduction, i.e. the printed
ttrcg store) with charge transfer switched on: Δ = 3 eV, U_pd − U_dd = 1 eV,
hybridisation in every channel, a crystal field, and in D4h a tetragonal one.
The from-scratch template gets the same per-configuration atomic parameters,
read off the fixture HAMILTONIAN blocks, and the same Δ, u, V(Γ) and
10Dq/Dt/Ds. Ground and final levels, stick intensities and the broadened
spectrum must agree.

Integer J only (V³⁺ d², Fe²⁺ d⁶, Ni²⁺ d⁸ in Oh; nid8ct, Ni²⁺ d⁸ in D4h).
Final levels of nid8ct agree to 2e-4 eV only: its store prints the 2p⁵ blocks
around E_av = 860 eV with fewer decimals, and its exactly degenerate 2p⁵ L̲
levels scatter by ±9e-5 eV in the Fortran store itself (1e-6 in the Oh stores,
E_av = 20 eV).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.ct_generator import build_ct_ban, generate_ct_ledge_template, hybridisation_operator_vectors
from multitorch.api.calc import _cache_ban, preload_fixture
from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store
from multitorch.spectrum.broaden import pseudo_voigt
from multitorch.spectrum.parity import spectral_parity
from multitorch.spectrum.sticks import get_sticks_from_banresult

POOL = dict(T=80.0, max_gs=40)
DELTA, U = 3.0, 1.0
CASES = [
    # element, valence, sym, n, crystal field, hybridisation, final-level tolerance (eV)
    ("V", "iii", "oh", 2, {"tendq": 1.2}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Fe", "ii", "oh", 6, {"tendq": 1.0}, {"eg": 2.2, "t2g": 1.1}, 2e-5),
    ("Ni", "ii", "oh", 8, {"tendq": 1.1}, {"eg": 2.0, "t2g": 1.0}, 2e-5),
    ("Ni", "ii", "d4h", 8, {"tendq": 1.2, "dt": 0.05, "ds": 0.1}, {"b1": 2.0, "a1": 1.6, "b2": 1.0, "e": 0.8}, 2e-4),
]
IDS = [f"{c[0]}{c[1]}-{c[2]}" for c in CASES]
LABELS = (("gs", (2, "GROUND")), ("gs_lh", (2, "EXCITE")), ("ex", (3, "GROUND")), ("ex_lh", (3, "EXCITE")))


def _run(element, valence, sym, n, cf, hyb):
    fx = preload_fixture(element, valence, sym)
    fortran = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(fx.cowan_template, fx.decomposition, slater=0.8, soc=1.0),
        fx.rac, _cache_ban(fx, cf, DELTA, U, hyb, None))
    rac, store, dec = generate_ct_ledge_template(n, sym)
    for label, (section, kind) in LABELS:
        ours, theirs = dec.by_label(label), fx.decomposition.config(section, kind)
        values = theirs.parameter_values(0.8, 1.0)
        if ours.shells == theirs.shells:
            ours.reference = {name: float(values[name]) for name in ours.operators if name in values}
        else:   # nid8ct couples 3d before 2p in 2p^5 3d^9
            ours.reference = {ours.aliases()[alias]: float(values[name]) for alias, name in theirs.aliases().items()}
    scratch = assemble_and_diagonalize_in_memory(
        rebuild_hamiltonian_store(store, dec, slater=1.0, soc=1.0), rac,
        build_ct_ban(rac, sym, cf=cf, delta=DELTA, u=U, hybridisation=hyb))
    return fortran, scratch


def _levels(result, attr):
    E = torch.sort(torch.cat([getattr(t, attr) for t in {t.fs_sym if attr == "Ef" else t.gs_sym: t for t in result.triads}.values()]))[0]
    return E - E[0]


def _spectrum(E, M, hi):
    x = torch.linspace(-3.0, hi, 6000, dtype=DTYPE)
    return x, pseudo_voigt(x, E, M, fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.4, med_energy=0.5 * hi, mode="legacy")


@pytest.mark.parametrize("element,valence,sym,n,cf,hyb,tol", CASES, ids=IDS)
def test_charge_transfer_from_scratch_matches_fortran(element, valence, sym, n, cf, hyb, tol):
    fortran, scratch = _run(element, valence, sym, n, cf, hyb)

    gf, gs = _levels(fortran, "Eg"), _levels(scratch, "Eg")
    assert gf.shape == gs.shape
    assert float((gf - gs).abs().max()) < 1e-5
    ff, fs = _levels(fortran, "Ef"), _levels(scratch, "Ef")
    assert ff.shape == fs.shape
    assert float((ff - fs).abs().max()) < tol

    Ef, Mf = get_sticks_from_banresult(fortran, **POOL)[:2]
    Es, Ms = get_sticks_from_banresult(scratch, **POOL)[:2]
    # V3+: 3T1g levels within kT at 80 K, so Boltzmann weights magnify the 1e-6 eV level noise
    assert float(Ms.sum()) == pytest.approx(float(Mf.sum()), rel=1e-4)
    hi = float(max(Ef.max() - Ef.min(), Es.max() - Es.min())) + 3.0
    p = spectral_parity(*_spectrum(Es - Es.min(), Ms, hi), *_spectrum(Ef - Ef.min(), Mf, hi))
    assert p.cosine > 0.99999, p


def test_hybridisation_channels_project_onto_their_orbitals():
    """Analytic oracle for the route phases: on one d electron (unit reduced elements) each
    hybridisation actor is √5 times the projector onto its orbitals (b1 x²−y², a1 z², b2 xy,
    e xz/yz; Oh eg, t2g), and the channels sum to √5·1."""
    from multitorch.angular.rac_generator import _operator_real_matrix

    # real d harmonics in _c2r_unitary order m = -2..2: xy, yz, z², xz, x²−y²
    expected = {
        "d4h": {"B1HYBR": [0, 0, 0, 0, 1], "A1HYBR": [0, 0, 1, 0, 0], "B2HYBR": [1, 0, 0, 0, 0], "EHYBR": [0, 1, 0, 1, 0]},
        "oh": {"EGHYBR": [0, 0, 1, 0, 1], "T2GHYBR": [1, 1, 0, 1, 0]},
    }
    for sym, channels in expected.items():
        total = np.zeros((5, 5))
        for channel, diag in channels.items():
            M = sum(_operator_real_matrix(2, 2, k, v) for k, v in hybridisation_operator_vectors(channel, sym).items())
            np.testing.assert_allclose(M, math.sqrt(5.0) * np.diag(diag), atol=1e-12, err_msg=f"{sym} {channel}")
            total += M
        np.testing.assert_allclose(total, math.sqrt(5.0) * np.eye(5), atol=1e-12)


@pytest.mark.parametrize("seed", [2, 5])
def test_charge_transfer_from_scratch_is_independent_of_the_lapack_basis_choice(seed):
    """The Oh Fe2+ and D4h nid8ct oracles with every numpy eigen/singular basis scrambled (fresh process)."""
    import subprocess
    import sys
    from pathlib import Path

    tools = Path(__file__).parent.parent / "tools"
    code = f"""
import sys; sys.path.insert(0, {str(tools)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
import lapack_scramble; lapack_scramble.install({seed})
import test_ct_from_scratch_fortran_parity as t
for case in (t.CASES[1], t.CASES[3]):
    t.test_charge_transfer_from_scratch_matches_fortran(*case)
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
