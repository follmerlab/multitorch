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


def test_configurations_of_equal_dimensions_are_refused():
    """Inject the collision the dimension decoding cannot survive and watch it raise.

    ``_find_transi_block`` takes no configuration argument: it separates a
    triad's two configurations by ``(n_bra, n_ket)`` alone, so equal dimensions
    would silently give configuration 2 configuration 1's transition block.
    Every reachable ion keeps the pairs apart today (at n = 8 the final-manifold
    dims already coincide, and only the ground dims still differ), which is
    exactly why nothing had ever exercised the failure.
    """
    import dataclasses

    cache = preload_fixture("Ni", "ii", "oh")
    nconf = cache.ban.nconf_gs
    gs, fs = _get_config_dims_from_transi(cache.rac, nconf)
    assert any(len(v) == nconf and len(set(v)) > 1 for v in gs.values())

    dipoles = [b for b in cache.rac.blocks
               if b.kind == "TRANSI" and "HYBR" not in b.geometry and b.add_entries]
    first = dipoles[0]
    twin = [b for b in dipoles if (b.bra_sym, b.op_sym, b.ket_sym, b.geometry)
            == (first.bra_sym, first.op_sym, first.ket_sym, first.geometry) and b is not first]
    assert twin, "need a second configuration of this triad to collide with"

    blocks = [dataclasses.replace(b, n_bra=first.n_bra, n_ket=first.n_ket) if b is twin[0] else b
              for b in cache.rac.blocks]
    collided = dataclasses.replace(cache.rac, blocks=blocks)
    with pytest.raises(RuntimeError, match="share the TRANSI dimensions"):
        _get_config_dims_from_transi(collided, nconf)
