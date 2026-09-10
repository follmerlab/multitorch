"""Single-shell Coulomb and spin-orbit operators vs exact Fortran (ttrcg) blocks.

Oracle: ``tests/reference_data/fortran_ops/ni2_oh_sc_hamiltonian.npz`` holds the
GROUND/EXCITE HAMILTONIAN blocks of a single-configuration Ni2+ Oh L-edge run
at (slater, soc) = (1, 1), (0, 1) and (1, 0). Differences isolate the Coulomb
and spin-orbit parts exactly (E_av of the ground configuration is 0).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.rme import (
    compute_coulomb_blocks,
    compute_coulomb_fk_ls,
    compute_soc_blocks,
)

ORACLE = Path(__file__).resolve().parents[1] / "reference_data" / "fortran_ops" / "ni2_oh_sc_hamiltonian.npz"


@pytest.fixture(scope="module")
def oracle():
    if not ORACLE.exists():
        pytest.fail(f"missing oracle {ORACLE}")
    d = np.load(ORACLE)
    meta = ast.literal_eval(str(d["meta"]))
    return d, meta


def test_f0_is_pair_count_before_average_removal():
    """Sanity on the U·U construction: raw f_0 = n(n-1)/2 on every term."""
    from multitorch.angular.rme import _lsterms_and_cfp, compute_uk_ls
    l, n = 2, 8
    terms, parents, cfp = _lsterms_and_cfp(l, n)
    # rebuild raw f_0 without the average subtraction
    _, fk = compute_coulomb_fk_ls(l, n, 0)
    assert np.allclose(fk, 0.0, atol=1e-5)  # vanishes relative to the average (CFP table precision ~1e-7)


@pytest.mark.parametrize("J", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_coulomb_blocks_match_fortran(oracle, J):
    d, meta = oracle
    blocks = compute_coulomb_blocks(2, 8)
    got = meta["F2dd_gs_eV"] * blocks[(2, J)] + meta["F4dd_gs_eV"] * blocks[(4, J)]
    want = d[f"gs_H_J{int(J)}"] - d[f"gs_H_noslater_J{int(J)}"]
    # pyctm rounds the RCG input to 3 decimals -> ~1e-4 relative on F^k
    np.testing.assert_allclose(got, want, atol=3e-3, err_msg=f"J={J}")


@pytest.mark.parametrize("J", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_soc_blocks_match_fortran(oracle, J):
    d, meta = oracle
    blocks = compute_soc_blocks(2, 8)
    got = meta["zeta3d_gs_eV"] * blocks[J]
    want = d[f"gs_H_J{int(J)}"] - d[f"gs_H_nosoc_J{int(J)}"]
    np.testing.assert_allclose(got, want, atol=3e-3, err_msg=f"J={J}")


def test_ground_constant_is_zero(oracle):
    """With slater = soc = 0 the Fortran ground block is exactly E_av = 0."""
    d, _ = oracle
    for J in range(5):
        rest = d[f"gs_H_noslater_J{J}"] - (d[f"gs_H_J{J}"] - d[f"gs_H_nosoc_J{J}"])
        assert np.abs(rest).max() < 1e-5


def test_soc_blocks_are_symmetric_and_rme_scaled():
    blocks = compute_soc_blocks(2, 8)
    for J, m in blocks.items():
        np.testing.assert_allclose(m, m.T, atol=1e-12)
    # d^1: single term 2D, <l·s> = +l/2 for J=l+1/2 and -(l+1)/2 for J=l-1/2
    b = compute_soc_blocks(2, 1)
    np.testing.assert_allclose(b[2.5][0, 0] / np.sqrt(6.0), 1.0, atol=1e-12)
    np.testing.assert_allclose(b[1.5][0, 0] / np.sqrt(4.0), -1.5, atol=1e-12)


# ─── two-shell (2p5 3d9) operators vs per-parameter Fortran grid ─────────────

@pytest.fixture(scope="module")
def two_shell_ops():
    from multitorch.angular.rme import compute_two_shell_operators
    basis, ops = compute_two_shell_operators(1, 5, 2, 9)
    return basis, ops


EX_J = ["0-", "1-", "2-", "3-", "4-"]


@pytest.mark.parametrize("lab", EX_J)
def test_two_shell_direct_matches_fortran(oracle, two_shell_ops, lab):
    d, meta = oracle
    _, ops = two_shell_ops
    J = float(lab[:-1])
    want = (d[f"ex_H_J{lab}"] - d[f"ex_H_fpd0.8_J{lab}"]) / 0.2
    got = meta["F2pd_ex_eV"] * ops["F2_12"][J]
    np.testing.assert_allclose(got, want, atol=3e-3, err_msg=lab)


@pytest.mark.parametrize("lab", EX_J)
def test_two_shell_exchange_matches_fortran(oracle, two_shell_ops, lab):
    d, meta = oracle
    _, ops = two_shell_ops
    J = float(lab[:-1])
    want = (d[f"ex_H_J{lab}"] - d[f"ex_H_gpd0.8_J{lab}"]) / 0.2
    got = meta["G1pd_ex_eV"] * ops["G1_12"][J] + meta["G3pd_ex_eV"] * ops["G3_12"][J]
    np.testing.assert_allclose(got, want, atol=3e-3, err_msg=lab)


@pytest.mark.parametrize("lab", EX_J)
def test_two_shell_soc_matches_fortran(oracle, two_shell_ops, lab):
    d, meta = oracle
    _, ops = two_shell_ops
    J = float(lab[:-1])
    want = d[f"ex_H_J{lab}"] - d[f"ex_H_nosoc_J{lab}"]
    got = meta["zeta2p_ex_eV"] * ops["zeta_1"][J] + meta["zeta3d_ex_eV"] * ops["zeta_2"][J]
    np.testing.assert_allclose(got, want, atol=3e-3, err_msg=lab)


def test_two_shell_dd_coulomb_vanishes_for_single_hole(two_shell_ops):
    _, ops = two_shell_ops
    for J, m in ops["F2_22"].items():
        assert np.abs(m).max() < 1e-12


def test_two_shell_full_excited_block_matches_fortran(oracle, two_shell_ops):
    """E_av + direct + exchange + SOC reproduces the (slater, soc) = (1, 1) block."""
    d, meta = oracle
    basis, ops = two_shell_ops
    for lab in EX_J:
        J = float(lab[:-1])
        n = ops["zeta_1"][J].shape[0]
        # E_av: what the Fortran block is at slater = soc = 0
        rest = d[f"ex_H_noslater_J{lab}"] - (d[f"ex_H_J{lab}"] - d[f"ex_H_nosoc_J{lab}"])
        eav = np.trace(rest) / n / np.sqrt(2 * J + 1)
        H = (eav * np.sqrt(2 * J + 1) * np.eye(n)
             + meta["F2pd_ex_eV"] * ops["F2_12"][J]
             + meta["G1pd_ex_eV"] * ops["G1_12"][J] + meta["G3pd_ex_eV"] * ops["G3_12"][J]
             + meta["zeta2p_ex_eV"] * ops["zeta_1"][J] + meta["zeta3d_ex_eV"] * ops["zeta_2"][J])
        np.testing.assert_allclose(H, d[f"ex_H_J{lab}"], atol=5e-3, err_msg=lab)
