"""End-to-end parity of the from-scratch pipeline against the Fortran chain.

The bundled ``ni2_d8_oh`` fixture is a pyctm/ttmult run at 80 % Slater
reduction (verified 2026-09-09 against regenerated runs, max |Δ| ≤ 2e-5).
Feeding the from-scratch generator the *same* atomic parameters (the RCG
input values of that run) must reproduce the fixture-path spectrum with the
ligand-hole configuration decoupled (lmct = 0): eigenvalues, transition
matrices and stick intensities. Before 2026-09-10 the from-scratch path was
at cosine 0.47 (audit S3/S4/S5/S7).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.rac_generator import generate_ledge_rac
from multitorch.api.calc import _build_ban_from_rac, _run_phase5_pipeline
from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
from multitorch.spectrum.broaden import pseudo_voigt
from multitorch.spectrum.sticks import get_sticks_from_banresult

RY = 13.605693122994

# RCG input of the ni2_d8_oh run (pyctm writes the reduced values to 3 decimals).
NI_80PCT = dict(
    raw_slater_gs_ry={"F2": 9.787 / RY, "F4": 6.078 / RY},
    raw_zeta_gs_ry=0.083 / RY,
    raw_slater_ex_ry={"F2_pd": 6.177 / RY, "G1_pd": 4.630 / RY, "G3_pd": 2.633 / RY,
                      "F2_dd": 0.0, "F4_dd": 0.0},
    raw_zeta_ex_ry={"p": 11.507 / RY, "d": 0.102 / RY},
)


def _sticks_fixture():
    res = _run_phase5_pipeline("Ni", "ii", "oh", "l", {"tendq": 1.0},
                               slater=1.0, soc=1.0, delta=100.0, lmct=0.0, mlct=None)
    E, M, _ = get_sticks_from_banresult(res, T=80.0, max_gs=1)
    keep = E < E.min() + 40.0          # drop the decoupled ligand-hole final states
    return E[keep], M[keep]


def _sticks_from_scratch(sym):
    rac, cowan = generate_ledge_rac(2, 8, sym=sym, **NI_80PCT)
    ban = _build_ban_from_rac(rac, tendq=1.0, dt=0.0, ds=0.0, sym=sym)
    res = assemble_and_diagonalize_in_memory(cowan, rac, ban)
    return get_sticks_from_banresult(res, T=80.0, max_gs=1)[:2]


def _spectrum(E, M):
    x = torch.linspace(float(E.min()) - 3, float(E.max()) + 3, 4000, dtype=DTYPE)
    y = pseudo_voigt(x, E, M, fwhm_g=0.2, fwhm_l=0.2, fwhm_l2=0.4,
                     med_energy=0.5 * float(E.min() + E.max()), mode="legacy")
    return y / y.max()


@pytest.mark.parametrize("sym", ["oh", "d4h"])
def test_from_scratch_ni_d8_matches_fortran_fixture(sym):
    Ef, Mf = _sticks_fixture()
    Es, Ms = _sticks_from_scratch(sym)
    # stick energies relative to the lowest one, matched exactly
    ef = np.round(Ef.numpy() - Ef.numpy().min(), 4)
    es = np.round(Es.numpy() - Es.numpy().min(), 4)
    mf = Mf.numpy() / Mf.numpy().sum()
    ms = Ms.numpy() / Ms.numpy().sum()
    strong = {e for e, m in zip(ef, mf) if m > 1e-4}
    assert strong <= set(es), sorted(strong - set(es))
    for e in strong:
        assert abs(ms[es == e].sum() - mf[ef == e].sum()) < 1e-5, e
    # broadened spectrum on a common relative grid
    ys = _spectrum(Es - Es.min(), Ms)
    yf = _spectrum(Ef - Ef.min(), Mf)
    cos = float((ys @ yf) / (ys.norm() * yf.norm()))
    assert cos > 0.99999, cos
