"""Ligand-to-metal hopping operators vs the Fortran ttrcg stores (WP-C, increment C1a).

Oracle: the TRANSITION MULTIPOLE blocks of section 2 (ground manifold: metal
d^n and ligand-hole d^(n+1)L) of every bundled two-configuration fixture, which
ttrcg writes for the spin-scalar one-electron transfer at orbital ranks 0, 2, 4.
Elementwise agreement to the store's 6-decimal print precision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.cowan_operators import hopping_blocks
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
