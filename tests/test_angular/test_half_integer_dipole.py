"""Half-integer TRANSI blocks against the Fortran RAC (WP-B2a-3, oracle 2).

The dipole of an odd-electron ion is emitted one block per invariant of the
triad (Γ_bra, T1u, Γ_ket): Γ8 ⊗ T1u ⊗ Γ8 carries two, every other triad one,
which is exactly the pattern ttrac writes in the .ban (``S1+ 1- S1- 0`` and
``... 1``). What can be compared with ttrac's numbers is what the gauge leaves
invariant:

* the triad count per (Γ_bra, Γ_ket) — the number of invariants;
* Σ c²·n_bra·n_ket summed over the blocks of a triad, and its partial sums over
  one J row-block or column-block. A copy basis of a (J, Γ) sector is a gauge
  (an orthogonal mixing carried by the Hamiltonian, the crystal field and the
  dipole alike), and so is a sign per (J, Γ, copy), so individual coefficients
  are not comparable; sums over a whole J sector are.

The individual channels of a two-invariant triad are not comparable either —
the channel basis is free up to an orthogonal rotation, which leaves the
intensity the assembler sums over the .ban's repeated triads unchanged — so
only their sum is checked. Spectra are checked in
tests/test_integration/test_from_scratch_fortran_parity.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from multitorch.angular.point_group import oh_branching
from multitorch.angular.rac_generator import (
    _get_excited_j_sizes, _get_terms, _j_sector_sizes, generate_ledge_template,
)
from multitorch.io.read_rme import read_rme_rac_full

REFDATA = Path(__file__).parent.parent / "reference_data"
# ttrac spells the double-group irreps S0/S1/^S0, the generator 1/2, 3/2, ^1/2
BUTLER_TO_FORTRAN = {"1/2": "S0", "3/2": "S1", "^1/2": "^S0"}
BUTLER_TO_IRREP = {"1/2": "E1/2", "3/2": "G3/2", "^1/2": "E5/2"}
IONS = [("cr3_d3_oh", 3), ("mn2_d5_oh", 5), ("fe3_d5_oh", 5), ("co2_d7_oh", 7)]


def _positions(j_sizes, irrep):
    """Block start position (1-based, as the ADD entries count) -> J."""
    out, pos = {}, 1
    for J in sorted(j_sizes):
        for _ in range(oh_branching(J).get(irrep, 0)):
            out[pos] = J
            pos += j_sizes[J]
    return out


def _weights(blocks, rowpos, colpos):
    """(total, per J_bra, per J_ket) of Σ c²·n_bra·n_ket over a triad's blocks.

    ttrac writes some ADD entries with a zero coefficient; they are dropped so
    the J sectors present on both sides are the ones that carry intensity.
    """
    row, col, total = {}, {}, 0.0
    for b in blocks:
        for e in b.add_entries:
            if abs(e.coeff) < 1e-12:
                continue
            w = e.coeff ** 2 * e.nbra * e.nket
            row[rowpos[e.bra]] = row.get(rowpos[e.bra], 0.0) + w
            col[colpos[e.ket]] = col.get(colpos[e.ket], 0.0) + w
            total += w
    return total, row, col


@pytest.fixture(scope="module")
def emitted():
    return {n: generate_ledge_template(2, n, sym="oh")[0] for _, n in IONS}


@pytest.mark.parametrize("name,n", IONS, ids=[i[0] for i in IONS])
def test_transi_blocks_match_the_fortran_rac(name, n, emitted):
    gs_j_sizes = _j_sector_sizes(_get_terms(2, n))
    ex_j_sizes = _get_excited_j_sizes(2, n, 1, 6)

    ours = {}
    for b in emitted[n].blocks:
        if b.kind == "TRANSI":
            assert b.geometry == "MULTIPOLE" and b.op_sym == "1-"
            ours.setdefault((b.bra_sym, b.ket_sym), []).append(b)

    # the fixtures are two-configuration; the blocks of the d^n configuration we
    # generate are the ones of our dimensions, as the assembler selects them
    dims = {(b.bra_sym, b.ket_sym): (b.n_bra, b.n_ket) for bs in ours.values() for b in bs}
    fixture = read_rme_rac_full(REFDATA / name / f"{name}.rme_rac")
    theirs = {}
    for b in fixture.blocks:
        if b.kind != "TRANSI" or b.geometry != "MULTIPOLE" or not b.add_entries:
            continue
        for key, (n_bra, n_ket) in dims.items():
            if (BUTLER_TO_FORTRAN[key[0][:-1]] + "+", BUTLER_TO_FORTRAN[key[1][:-1]] + "-") \
                    == (b.bra_sym, b.ket_sym) and (b.n_bra, b.n_ket) == (n_bra, n_ket):
                theirs.setdefault((b.bra_sym, b.ket_sym), []).append(b)

    assert {(BUTLER_TO_FORTRAN[g[:-1]] + "+", BUTLER_TO_FORTRAN[e[:-1]] + "-")
            for g, e in ours} == set(theirs)

    for key, blocks in sorted(ours.items()):
        fkey = (BUTLER_TO_FORTRAN[key[0][:-1]] + "+", BUTLER_TO_FORTRAN[key[1][:-1]] + "-")
        gpos = _positions(gs_j_sizes, BUTLER_TO_IRREP[key[0][:-1]])
        epos = _positions(ex_j_sizes, BUTLER_TO_IRREP[key[1][:-1]])
        # one block per invariant: two only where Γ8 ⊗ T1u ⊃ 2 Γ8
        assert len(blocks) == len(theirs[fkey]), key
        assert len(blocks) == (2 if key == ("3/2+", "3/2-") else 1), key

        total, row, col = _weights(blocks, gpos, epos)
        f_total, f_row, f_col = _weights(theirs[fkey], gpos, epos)
        assert total == pytest.approx(f_total, rel=2e-6), key
        for tag, a, b in (("J_bra", row, f_row), ("J_ket", col, f_col)):
            assert sorted(a) == sorted(b), (key, tag)
            for J in sorted(a):
                assert a[J] == pytest.approx(b[J], rel=2e-6), (key, tag, J)
