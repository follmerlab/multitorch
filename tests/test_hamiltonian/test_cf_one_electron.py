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
