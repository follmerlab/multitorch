"""Configuration operators in the Fortran store basis (``angular/cowan_operators.py``).

Oracle for the basis gauge: the SHELL2 (U^k of the 3d shell) blocks that ttrcg
wrote into the bundled stores. Our CFP-derived SHELL blocks equal them after
the term phase σ = (−1)^(L+S−S_min), for every multi-term d^n; the same gauge
makes the HAMILTONIAN decomposition exact (``test_build_cowan.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.cowan_operators import configuration_operators
from multitorch.angular.rme import (
    _j_basis_for_terms,
    _lsterms_and_cfp,
    compute_shell_blocks,
    compute_uk_ls,
)
from multitorch.hamiltonian.build_cowan import j_value, read_cowan_metadata
from multitorch.io.read_rme import read_cowan_store

REFDATA = Path(__file__).parent.parent / "reference_data"


@pytest.mark.parametrize("name,n", [("v3_d2_oh", 2), ("cr3_d3_oh", 3), ("mn2_d5_oh", 5),
                                    ("fe2_d6_oh", 6), ("co2_d7_oh", 7)])
def test_term_gauge_maps_shell_blocks_onto_fortran(name, n):
    rcg = REFDATA / name / f"{name}.rme_rcg"
    meta, store = read_cowan_metadata(rcg)[2], read_cowan_store(rcg)[2]
    terms, parents, cfp = _lsterms_and_cfp(2, n)
    basis = _j_basis_for_terms(terms)
    sigma = {J: np.array([(-1) ** int(round(s.ls_term.L + s.ls_term.S - (n % 2) / 2)) for s in st])
             for J, st in basis.items()}
    checked = 0
    for k in (2, 4):
        ours = compute_shell_blocks(2, n, k, terms, compute_uk_ls(2, n, k, terms, parents, cfp))
        for m, M in zip(meta, store):
            if not (m.operator.startswith("SHELL") and m.block_type == "GROUND" and m.op_sym == f"{k}+"):
                continue
            Jb, Jk = j_value(m.bra_sym), j_value(m.ket_sym)
            O = ours.get((Jb, Jk), np.zeros(tuple(M.shape)))
            np.testing.assert_allclose(sigma[Jb][:, None] * O * sigma[Jk][None, :], M.numpy(), atol=5e-6)
            checked += 1
    assert checked > 10


def test_operator_names_by_shell_count():
    assert set(configuration_operators(((2, 8),)).blocks) == {"F2_11", "F4_11", "zeta_1"}
    two = configuration_operators(((1, 5), (2, 9)))
    assert {"F2_12", "G1_12", "G3_12", "zeta_1", "zeta_2"} <= set(two.blocks)
    three = configuration_operators(((1, 5), (2, 8), (2, 9)))
    assert {"F2_22", "F4_22", "F2_12", "G1_12", "G3_12", "zeta_1", "zeta_2"} <= set(three.blocks)
    assert configuration_operators(()).dims == {0.0: 1}


def test_three_shell_dimensions_match_fortran():
    """p⁵ d⁸ L̲⁹ J-block sizes equal the fe2_d6_oh section-3 EXCITE blocks."""
    rcg = REFDATA / "fe2_d6_oh" / "fe2_d6_oh.rme_rcg"
    meta, store = read_cowan_metadata(rcg)[3], read_cowan_store(rcg)[3]
    fortran = {j_value(m.bra_sym): M.shape[0] for m, M in zip(meta, store)
               if m.operator == "HAMILTONIAN" and m.block_type == "EXCITE"}
    assert configuration_operators(((1, 5), (2, 8), (2, 9))).dims == fortran


def test_spectator_shell_leaves_spectrum_of_two_shell_operator():
    """ζ_2p on p⁵ d⁹ L̲⁹ has the eigenvalues of ζ_2p on p⁵ d⁹, each repeated over the 10 L̲ states."""
    import math
    two = configuration_operators(((1, 5), (2, 9)))
    three = configuration_operators(((1, 5), (2, 9), (2, 9)))
    for name in ("zeta_1", "zeta_2", "G1_12", "F2_12"):
        ev2 = sorted(np.concatenate([
            np.repeat(np.linalg.eigvalsh(M / math.sqrt(2 * J + 1)), int(2 * J + 1))
            for J, M in two.blocks[name].items()]))
        ev3 = sorted(np.concatenate([
            np.repeat(np.linalg.eigvalsh(M / math.sqrt(2 * J + 1)), int(2 * J + 1))
            for J, M in three.blocks[name].items()]))
        assert len(ev3) == 10 * len(ev2)
        np.testing.assert_allclose(ev3, np.repeat(ev2, 10), atol=1e-9, err_msg=name)


def test_rejects_closed_shells():
    with pytest.raises(ValueError):
        configuration_operators(((2, 10),))
