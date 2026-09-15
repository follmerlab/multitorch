"""Ligand-to-metal hopping operators vs the Fortran ttrcg stores (WP-C, increments C1a, C1b).

Oracle: the TRANSITION MULTIPOLE blocks that ttrcg writes for the spin-scalar
one-electron transfer at orbital ranks 0, 2, 4 in every bundled
two-configuration fixture — section 2 (ground manifold: d^n and d^(n+1)L) and
section 3 (final manifold under the 2p hole: 2p^5 d^(n+1) and 2p^5 d^(n+2)L).
Elementwise agreement to the store's 6-decimal print precision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.cowan_operators import _store_basis, final_state_hopping_blocks, hopping_blocks
from multitorch.hamiltonian.build_cowan import read_cowan_configurations, read_cowan_metadata, j_value
from multitorch.io.read_rme import read_cowan_store

REFDATA = Path(__file__).parent.parent / "reference_data"
STORES = ["v3_d2_oh", "cr3_d3_oh", "mn2_d5_oh", "fe3_d5_oh", "fe2_d6_oh", "co2_d7_oh", "ni2_d8_oh", "nid8ct"]


@pytest.mark.parametrize("name", STORES)
def test_ground_manifold_hopping_matches_ttrcg(name):
    rcg = REFDATA / name / f"{name}.rme_rcg"
    store, meta, confs = read_cowan_store(rcg), read_cowan_metadata(rcg), read_cowan_configurations(rcg)
    (l, n), = confs[2]["GROUND"]
    assert confs[2]["EXCITE"][0] == (l, n + 1) and confs[2]["EXCITE"][-1] == (l, 4 * l + 1)
    checked = 0
    for j, m in enumerate(meta[2]):
        if m.operator != "MULTIPOLE":
            continue
        rank = int(m.op_sym.strip("^+-"))
        ours = hopping_blocks(n, rank, l)[(j_value(m.bra_sym), j_value(m.ket_sym))]
        fortran = store[2][j].numpy()
        assert ours.shape == fortran.shape
        np.testing.assert_allclose(ours, fortran, atol=2e-6, err_msg=f"{name} rank {rank} {m.bra_sym}->{m.ket_sym}")
        checked += 1
    assert checked >= 20


def _to_core_first(shells):
    """Rows of a (valence, core) two-shell store basis as ± rows of the (core, valence) basis.

    Swapping the coupling order of (a b) S L costs (-1)^(S_a+S_b-S + L_a+L_b-L),
    and anticommuting the two shells' electrons (-1)^(n_a n_b); the term gauge is
    a product over shells and does not change.
    """
    swapped = _store_basis(shells[::-1])
    out = {}
    for J, states in _store_basis(shells).items():
        index = {(tuple(t.index for t in terms), S, L): i for i, (terms, _, S, L) in enumerate(swapped[J])}
        perm, sign = [], []
        for (a, b), _, S, L in states:
            perm.append(index[((b.index, a.index), S, L)])
            sign.append((-1.0) ** round(a.S + b.S - S + a.L + b.L - L + shells[0][1] * shells[1][1]))
        out[J] = (np.array(perm), np.array(sign))
    return out


@pytest.mark.parametrize("name", STORES)
def test_final_manifold_hopping_matches_ttrcg(name):
    rcg = REFDATA / name / f"{name}.rme_rcg"
    store, meta, confs = read_cowan_store(rcg), read_cowan_metadata(rcg), read_cowan_configurations(rcg)
    (l, n), = confs[2]["GROUND"]
    reorder = None
    if confs[3]["GROUND"] == ((l, n + 1), (1, 5)):   # nid8ct couples the valence shell first
        reorder = _to_core_first(confs[3]["GROUND"])
    else:
        assert confs[3]["GROUND"] == ((1, 5), (l, n + 1))
    assert confs[3]["EXCITE"][0] == (1, 5) and confs[3]["EXCITE"][-1] == (l, 4 * l + 1)
    checked = 0
    for j, m in enumerate(meta[3]):
        if m.operator != "MULTIPOLE":
            continue
        rank = int(m.op_sym.strip("^+-"))
        Jb = j_value(m.bra_sym)
        ours = final_state_hopping_blocks(n, rank, l)[(Jb, j_value(m.ket_sym))]
        if reorder is not None:
            perm, sign = reorder[Jb]
            ours = sign[:, None] * ours[perm]
        fortran = store[3][j].numpy()
        assert ours.shape == fortran.shape
        np.testing.assert_allclose(ours, fortran, atol=2e-6, err_msg=f"{name} rank {rank} {m.bra_sym}->{m.ket_sym}")
        checked += 1
    assert checked >= 20
