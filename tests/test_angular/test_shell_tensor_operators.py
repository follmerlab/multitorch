"""Metal-shell orbital tensors vs the Fortran ttrcg stores (WP-C, increment C2).

Oracle: the SHELL blocks (crystal-field operators U^k, k = 0, 2, 4 on the
metal d shell) of sections 2 and 3 of every bundled two-configuration fixture:
the metal and ligand-hole ground configurations d^n / d^(n+1)L, and the final
configurations 2p^5 d^(n+1) / 2p^5 d^(n+2)L — one, two and three open shells,
with the metal shell first or second. Elementwise to ttrcg's single precision
(rtol 1e-6: the U^0 diagonals reach 13 and differ by 6e-6).
Where the metal shell is closed (Ni d^8: 2p^5 3d^10 L) ttrcg writes zeros.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multitorch.angular.cowan_operators import shell_tensor_blocks
from multitorch.hamiltonian.build_cowan import read_cowan_configurations, read_cowan_metadata, j_value
from multitorch.io.read_rme import read_cowan_store

REFDATA = Path(__file__).parent.parent / "reference_data"
STORES = ["v3_d2_oh", "cr3_d3_oh", "mn2_d5_oh", "fe3_d5_oh", "fe2_d6_oh", "co2_d7_oh", "ni2_d8_oh", "nid8ct"]


def _metal_shell(shells, block_type):
    """Index of the metal d shell; in the ligand-hole configuration the last d shell is the ligand."""
    candidates = shells[:-1] if block_type == "EXCITE" else shells
    return next((i for i, (l, _) in enumerate(candidates) if l == 2), None)


@pytest.mark.parametrize("name", STORES)
def test_metal_shell_tensors_match_ttrcg(name):
    rcg = REFDATA / name / f"{name}.rme_rcg"
    store, meta, confs = read_cowan_store(rcg), read_cowan_metadata(rcg), read_cowan_configurations(rcg)
    checked, closed = set(), 0
    for sec in (2, 3):
        for j, m in enumerate(meta[sec]):
            if not m.operator.startswith("SHELL"):
                continue
            shells = confs[sec][m.block_type]
            fortran = store[sec][j].numpy()
            metal = _metal_shell(shells, m.block_type)
            if metal is None:
                np.testing.assert_allclose(fortran, 0.0, atol=2e-6)
                closed += 1
                continue
            rank = int(m.op_sym.strip("^+-"))
            ours = shell_tensor_blocks(shells, metal, rank)[(j_value(m.bra_sym), j_value(m.ket_sym))]
            assert ours.shape == fortran.shape
            np.testing.assert_allclose(ours, fortran, rtol=1e-6, atol=2e-6,
                                       err_msg=f"{name} sec{sec} {m.block_type} U{rank} {m.bra_sym}->{m.ket_sym}")
            checked.add((sec, m.block_type, len(shells), metal))
    # every (section, configuration, open-shell count, metal position) seen; a d^8 ion's 2p^5 3d^10 L has a closed metal
    d10_final = confs[2]["GROUND"][0][1] == 8
    assert len(checked) == (3 if d10_final else 4), checked
    assert (closed > 0) == d10_final
