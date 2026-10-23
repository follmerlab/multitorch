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

import math
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


# ─────────────────────────────────────────────────────────────
# Where the dipole gets its double-irrep dimension (residual 63)
# ─────────────────────────────────────────────────────────────

def test_the_dipole_does_not_consult_the_double_irrep_dimension_lookup():
    """The dipole is independent of ``target_double_irrep_dim``; the operators are not.

    Residual 62 forced that lookup to 1 for D4h and found every operator
    coefficient scaled by exactly sqrt(1/2) while PARA and PERP did not move at
    all, and recorded the stillness as a **caveat**: it showed the dipole never
    reads the lookup, so it said nothing about whether the dipole is right for
    a one-dimensional double irrep. This pins the independence as a fact rather
    than an observation, because it is what makes the dipole dimension-general:
    the emitter reads each extent off the partner tensor it is handed
    (``T[bra_extent, n_op, ket_extent]``) and the dimension enters through
    ``||T||_F``, so there is no lookup to get wrong.

    Dimension 1 is therefore not a new regime for the dipole emitter, and
    :func:`test_the_invariant_projection_does_not_degenerate_at_one_dimension`
    checks the projection that consumes ``T``.
    """
    from unittest import mock

    from multitorch.angular import double_group

    def coeffs(rac, kinds, geometries):
        return [e.coeff for b in rac.blocks
                if b.kind in kinds and b.geometry in geometries
                for e in b.add_entries]

    real = generate_ledge_template(2, 7, sym="d4h")[0]
    with mock.patch.object(double_group, "target_double_irrep_dim", return_value=1):
        forced = generate_ledge_template(2, 7, sym="d4h")[0]

    dip_real = coeffs(real, {"TRANSI"}, {"MULTIPOLE", "PARA", "PERP"})
    dip_forced = coeffs(forced, {"TRANSI"}, {"MULTIPOLE", "PARA", "PERP"})
    assert dip_real and dip_real == dip_forced, "the dipole read the dimension lookup"

    ops = {"HAMILTONIAN", "10DQ", "DT", "DS"}
    op_real = coeffs(real, {"GROUND", "EXCITE"}, ops)
    op_forced = coeffs(forced, {"GROUND", "EXCITE"}, ops)
    assert len(op_real) == len(op_forced) and op_real
    # Every operator coefficient carries sqrt(dim), so forcing dim 2 -> 1
    # scales all of them by one factor. If this stops holding, the lookup has
    # acquired a second consumer and the dipole half of the test is no longer
    # measuring independence.
    ratios = {round(f / r, 12) for r, f in zip(op_real, op_forced) if abs(r) > 1e-12}
    assert ratios == {round(math.sqrt(0.5), 12)}, sorted(ratios)


def test_the_invariant_projection_does_not_degenerate_at_one_dimension():
    """``real_invariant_coefficients`` at a 1-D double irrep, which D3d has and D4h does not.

    D3's double group has two **one-dimensional** spinor irreps (residual 60),
    so a trigonal half-integer dipole will hand this function partner tensors
    of extent 1 where every D4h one is 2 and Oh's are 2 or 4. The projection is
    shape-agnostic -- a rank-1 family gives one channel whose coefficients are
    the tensors' Frobenius norms -- and that is checked here at extent 1
    alongside the two extents the Fortran path already validates, so the
    trigonal case is not resting on the claim alone.

    Not a physics oracle: it pins an algebraic property of the projection, and
    the physics of the trigonal dipole still needs the descent limit.
    """
    import numpy as np

    from multitorch.angular.double_group import real_invariant_coefficients

    rng = np.random.default_rng(0)
    for shape in [(1, 1, 1), (1, 3, 1), (1, 3, 2), (2, 3, 2), (4, 3, 4)]:
        base = rng.normal(size=shape)
        scales = (1.0, -0.7, 0.3)
        R = real_invariant_coefficients([c * base for c in scales])
        assert R.shape == (len(scales), 1), (shape, R.shape)
        assert np.allclose(np.abs(R[:, 0]),
                           [abs(c) * np.linalg.norm(base) for c in scales]), shape


def test_target_partners_is_where_a_new_group_enters_the_half_integer_path():
    """The dimension gate residual 62 did **not** widen.

    Residual 62 widened ``target_double_irrep_dim``, which supplies the
    operator coefficient's ``sqrt(dim)``. The partner *slicer* is a different
    function and still carries the tetragonal table: ``target_partners`` slices
    with ``D4H_DOUBLE_IRREP_DIM`` and calls ``oh_to_d4h_double_subduction``
    directly. So the two dimensions a trigonal calculation needs would come
    from two places, only one of them general.

    It refuses rather than going quietly wrong, and this test says so, so that
    whoever builds the D3d double subduction finds the seam named instead of
    inferring it from residual 62's title.
    """
    from multitorch.angular.double_group import gauged_basis

    basis = gauged_basis([("gs", [0.5, 1.5]), ("ex", [0.5, 1.5])], d4h=True)
    node = next(n for n in basis.nodes if n[2] == "E1/2")
    basis.target_partners(node, "E1/2", target_group="D4h")      # works today
    with pytest.raises(ValueError, match="unknown subduction target"):
        basis.target_partners(node, "E1/2", target_group="D3d")
