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
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from multitorch.api.calc import _build_ban_from_rac, _run_phase5_pipeline


def ballhausen_d8_levels(tendq: float, dt: float = 0.0, ds: float = 0.0) -> np.ndarray:
    """Sorted unique two-hole levels (relative to the lowest) for d^8."""
    dq = tendq / 10.0
    e = np.array([
        6 * dq - 2 * ds - 6 * dt,   # z2
        6 * dq + 2 * ds - dt,       # x2-y2
        -4 * dq + 2 * ds - dt,      # xy
        -4 * dq - ds + 4 * dt,      # xz
        -4 * dq - ds + 4 * dt,      # yz
    ])
    holes = -e
    pairs = [holes[i] + holes[j] for i, j in itertools.combinations_with_replacement(range(5), 2)]
    lv = np.unique(np.round(np.array(pairs), 9))
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


def _from_scratch_ground_levels(sym: str, cf: dict) -> np.ndarray:
    from multitorch.angular.rac_generator import generate_ledge_rac
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory

    rac, cowan = generate_ledge_rac(
        l_val=2, n_val_gs=8,
        raw_slater_gs_ry={"F0": 0.0, "F2": 0.0, "F4": 0.0},
        raw_zeta_gs_ry=0.0,
        raw_slater_ex_ry={"F2_dd": 0.0, "F4_dd": 0.0, "G1_pd": 0.0,
                          "G3_pd": 0.0, "F2_pd": 0.0},
        raw_zeta_ex_ry={"d": 0.0, "p": 0.0},
        sym=sym,
    )
    ban = _build_ban_from_rac(
        rac, tendq=cf.get("tendq", 0.0), dt=cf.get("dt", 0.0),
        ds=cf.get("ds", 0.0), sym=sym,
    )
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
