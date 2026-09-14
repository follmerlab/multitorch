"""Per-configuration block dimensions against the Fortran ttban output.

Oracle: every ``TRANSFORMED MATRIX FOR TRIAD ( gs op fs ) (1* N)`` header in a
bundled ``.ban_out`` prints N, the number of final states ttban diagonalised for
that final-state irrep. Before 2026-09-14 the assembler counted each PRMULT copy
of a triad's transition block as another configuration, duplicating the
ligand-hole configuration in the Γ8 irreps of every half-integer-J fixture
(Fe3+ S1-: 3810 states instead of 1410).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from multitorch.api.calc import preload_fixture
from multitorch.hamiltonian.assemble import _get_config_dims_from_transi

REFDATA = Path(__file__).parent.parent / "reference_data"
HEADER = re.compile(r"TRIAD \( (\S+)\s+(\S+)\s+(\S+)\s+\) \(1\*\s+(\d+)\)")

CASES = [("ti4_d0_oh", "Ti", "iv"), ("v3_d2_oh", "V", "iii"), ("cr3_d3_oh", "Cr", "iii"),
         ("mn2_d5_oh", "Mn", "ii"), ("fe3_d5_oh", "Fe", "iii"), ("fe2_d6_oh", "Fe", "ii"),
         ("co2_d7_oh", "Co", "ii"), ("ni2_d8_oh", "Ni", "ii")]


@pytest.mark.parametrize("case,element,valence", CASES, ids=[c[0] for c in CASES])
def test_final_state_dimensions_match_ttban(case, element, valence):
    cache = preload_fixture(element, valence, "oh")
    gs, fs = _get_config_dims_from_transi(cache.rac, cache.ban.nconf_gs)
    fortran = {m.group(3): int(m.group(4)) for m in HEADER.finditer((REFDATA / case / f"{case}.ban_out").read_text())}
    assert fortran
    for fs_sym, n in fortran.items():
        assert sum(fs[fs_sym]) == n, (fs_sym, fs[fs_sym], n)
    assert all(len(v) <= cache.ban.nconf_gs for v in list(gs.values()) + list(fs.values()))
