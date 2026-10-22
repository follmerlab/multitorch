"""Tests for the Fortran-free RAC generator.

Validates that the generator produces ADD coefficients matching the
Fortran fixtures for all integer-J Oh systems (d^2, d^4, d^6, d^8).

Half-integer-J systems (d^3, d^5, d^7) require double-group support
and are deferred to a later stage.
"""
import math
import os

import numpy as np
import pytest
import torch

from multitorch.angular.point_group import (
    OH_IRREP_DIM,
    oh_branching,
    oh_coupling_coefficients,
    oh_coupling_coefficients_full,
)
from multitorch.angular.rac_generator import generate_ground_state_rac
from multitorch.io.read_rme import read_rme_rac_full


REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


# ---------- Coefficient-level tests (multiplicity-1 cases) ----------

NI2_D8_K4_FIXTURE = {
    ('A1', 0, 4): 1.0954451100, ('A1', 4, 0): 1.0954451100,
    ('A1', 4, 4): 0.5235703100,
    ('T1', 1, 3): 1.0954451200, ('T1', 1, 4): 1.0954451100,
    ('T1', 3, 1): 1.0954451200, ('T1', 3, 3): 0.7006490500,
    ('T1', 3, 4): -0.6179143800, ('T1', 4, 1): -1.0954451100,
    ('T1', 4, 3): 0.6179143800, ('T1', 4, 4): 0.4534251900,
    ('E', 2, 2): 0.8485281400, ('E', 2, 4): 0.9341987300,
    ('E', 4, 2): 0.9341987300, ('E', 4, 4): 0.1057771800,
    ('T2', 2, 2): -0.6928203200, ('T2', 2, 3): 1.0954451200,
    ('T2', 2, 4): 0.5720775500, ('T2', 3, 2): -1.0954451200,
    ('T2', 3, 3): -0.2335496800, ('T2', 3, 4): -0.9045340300,
    ('T2', 4, 2): 0.5720775500, ('T2', 4, 3): 0.9045340300,
    ('T2', 4, 4): -0.8420753600,
    ('A2', 3, 3): -0.8090398300,
}


def test_oh_coupling_k4_ni2_d8():
    """k=4 ADD coefficients match Ni d^8 Oh fixture (25/25)."""
    add = oh_coupling_coefficients(J_max=4, k=4, l=2)
    for (irrep, Jb, Jk), expected in NI2_D8_K4_FIXTURE.items():
        actual = add.get((irrep, Jb, Jk), 0.0)
        assert abs(actual - expected) < 1e-4, (
            f"{irrep} ({Jb},{Jk}): {actual:.10f} != {expected:.10f}"
        )


def test_oh_coupling_k0_identity():
    """k=0 ADD coefficients = sqrt(dim_Gamma/(2J+1))."""
    add = oh_coupling_coefficients(J_max=6, k=0, l=2)
    for (irrep, Jb, Jk), val in add.items():
        if Jb != Jk:
            assert False, f"k=0 should be diagonal, got ({irrep},{Jb},{Jk})"
        expected = math.sqrt(OH_IRREP_DIM[irrep] / (2 * Jb + 1))
        assert abs(val - expected) < 1e-8, (
            f"{irrep} J={Jb}: {val:.10f} != {expected:.10f}"
        )


def test_oh_coupling_k4_d6_extended():
    """k=4 with J_max=6 (d^6 range) includes J=6 entries correctly."""
    add = oh_coupling_coefficients(J_max=6, k=4, l=2)
    # A1, (4,6) should exist (BFS phase propagation)
    assert ('A1', 4, 6) in add
    assert abs(add[('A1', 4, 6)] - 0.5793654595) < 1e-4
    # A1, (6,6) should exist
    assert ('A1', 6, 6) in add
    assert abs(add[('A1', 6, 6)] - (-0.4040620566)) < 1e-4


# ---------- Generator structural tests ----------

@pytest.mark.parametrize("name,l,n", [
    ("ni2_d8_oh", 2, 8),
    ("v3_d2_oh", 2, 2),
])
def test_generator_coefficient_parity(name, l, n):
    """ADD coefficients match fixture for all mult=1 systems."""
    rac_file = os.path.join(REF_DIR, name, f"{name}.rme_rac")
    if not os.path.exists(rac_file):
        pytest.skip(f"No fixture: {rac_file}")

    fix_rac = read_rme_rac_full(rac_file)
    gen_rac, _ = generate_ground_state_rac(l=l, n=n)

    for fix_b in fix_rac.blocks:
        if fix_b.kind != 'GROUND' or not fix_b.bra_sym.endswith('+'):
            continue
        if fix_b.geometry not in ('HAMILTONIAN', '10DQ'):
            continue
        if not fix_b.add_entries or 'S' in fix_b.bra_sym:
            continue

        gen_b = None
        for gb in gen_rac.blocks:
            if (gb.kind == fix_b.kind and gb.bra_sym == fix_b.bra_sym
                    and gb.geometry == fix_b.geometry
                    and gb.n_bra == fix_b.n_bra and gb.add_entries):
                gen_b = gb
                break

        assert gen_b is not None, (
            f"Missing: {fix_b.bra_sym} {fix_b.geometry} {fix_b.n_bra}x{fix_b.n_ket}"
        )

        fix_lookup = {
            (e.bra, e.ket, e.nbra, e.nket): e.coeff
            for e in fix_b.add_entries
        }
        gen_lookup = {
            (e.bra, e.ket, e.nbra, e.nket): e.coeff
            for e in gen_b.add_entries
        }

        for pos, fix_coeff in fix_lookup.items():
            gen_coeff = gen_lookup.get(pos)
            assert gen_coeff is not None, (
                f"{fix_b.bra_sym} {fix_b.geometry}: missing entry at {pos}"
            )
            assert abs(gen_coeff - fix_coeff) < 1e-4, (
                f"{fix_b.bra_sym} {fix_b.geometry} {pos}: "
                f"{gen_coeff:.8f} != {fix_coeff:.8f}"
            )


@pytest.mark.parametrize("name,l,n", [
    ("ni2_d8_oh", 2, 8),
    ("v3_d2_oh", 2, 2),
    ("fe2_d6_oh", 2, 6),
])
def test_generator_block_dimensions(name, l, n):
    """Generated block dimensions match fixture for gerade irreps."""
    rac_file = os.path.join(REF_DIR, name, f"{name}.rme_rac")
    if not os.path.exists(rac_file):
        pytest.skip(f"No fixture: {rac_file}")

    fix_rac = read_rme_rac_full(rac_file)
    gen_rac, _ = generate_ground_state_rac(l=l, n=n)

    fix_dims = {}
    for b in fix_rac.blocks:
        if (b.kind == 'GROUND' and b.bra_sym.endswith('+')
                and b.geometry in ('HAMILTONIAN', '10DQ')
                and b.add_entries and 'S' not in b.bra_sym):
            fix_dims[(b.bra_sym, b.geometry)] = b.n_bra

    gen_dims = {}
    for b in gen_rac.blocks:
        if (b.kind == 'GROUND' and b.bra_sym.endswith('+')
                and b.geometry in ('HAMILTONIAN', '10DQ')
                and b.add_entries):
            gen_dims[(b.bra_sym, b.geometry)] = b.n_bra

    for key, fix_dim in fix_dims.items():
        gen_dim = gen_dims.get(key)
        assert gen_dim is not None, f"Missing block: {key}"
        assert gen_dim == fix_dim, (
            f"{key}: dim {gen_dim} != {fix_dim}"
        )


def test_generator_per_copy_coefficients():
    """Per-copy coupling coefficients are correct for mult=1."""
    # For mult=1, per-copy should equal the standard coefficient
    add_std = oh_coupling_coefficients(J_max=4, k=4, l=2)
    add_full = oh_coupling_coefficients_full(J_max=4, k=4, l=2)

    for (irrep, Jb, Jk), expected in add_std.items():
        actual = add_full.get((irrep, Jb, 0, Jk, 0), 0.0)
        assert abs(actual - expected) < 1e-8, (
            f"{irrep} ({Jb},{Jk}): full={actual:.10f} std={expected:.10f}"
        )


# ---------- Assembled eigenvalue tests (multiplicity > 1) ----------

def _assemble_irrep_hamiltonian(rac, cowan_sec, irrep_sym, operators, xham):
    """Assemble the ground-state Hamiltonian for one irrep.

    Returns eigenvalues sorted ascending, or None if the irrep is absent.
    """
    from multitorch.io.read_rme import assemble_matrix_from_adds

    # Find operator blocks for this irrep
    op_blocks = []
    for op in operators:
        match = None
        for b in rac.blocks:
            if (b.kind == 'GROUND' and b.bra_sym == irrep_sym
                    and b.geometry == op and b.add_entries):
                match = b
                break
        op_blocks.append(match)

    if all(b is None for b in op_blocks):
        return None

    # Get block dimension from the first non-None block
    d = next(b.n_bra for b in op_blocks if b is not None)

    H = torch.zeros(d, d, dtype=torch.float64)
    for blk, xv in zip(op_blocks, xham):
        if blk is not None and blk.add_entries and xv != 0.0:
            H += assemble_matrix_from_adds(blk.add_entries, cowan_sec, d, d, scale=xv)

    dim_gamma = OH_IRREP_DIM.get(
        next((k for k, v in {
            'A1': '0+', 'A2': '^0+', 'E': '2+', 'T1': '1+', 'T2': '^1+'
        }.items() if v == irrep_sym), irrep_sym), 1)
    H = 0.5 * (H + H.T) * (1.0 / math.sqrt(dim_gamma))

    evals = torch.linalg.eigvalsh(H)
    return evals


# Map Butler label -> irrep name for dimension lookup
_BUTLER_TO_IRREP = {'0+': 'A1', '^0+': 'A2', '2+': 'E', '1+': 'T1', '^1+': 'T2'}


@pytest.mark.parametrize("name,l,n", [
    ("ni2_d8_oh", 2, 8),
    ("v3_d2_oh", 2, 2),
    ("fe2_d6_oh", 2, 6),
])
def test_generator_assembled_eigenvalues(name, l, n):
    """CF-assembled eigenvalues match fixture for all gerade irreps.

    Uses only the 10DQ (crystal-field, k=4) operator, which tests
    cross-J coupling and multiplicity > 1 ADD coefficients. The
    HAMILTONIAN (k=0) operator is diagonal in J and requires Slater
    integrals to match the fixture's pre-assembled COWAN matrices,
    so it is excluded here. The CF SHELL_4 blocks are pure angular
    factors identical in both fixture and generator.

    This is the definitive validation for multiplicity > 1 cases
    (e.g. Fe d6) where per-copy ADD coefficients are basis-dependent
    but the assembled matrix is isospectral.
    """
    from multitorch.io.read_rme import read_cowan_store

    rac_file = os.path.join(REF_DIR, name, f"{name}.rme_rac")
    rcg_file = os.path.join(REF_DIR, name, f"{name}.rme_rcg")
    if not os.path.exists(rac_file) or not os.path.exists(rcg_file):
        pytest.skip(f"No fixture: {rac_file}")

    fix_rac = read_rme_rac_full(rac_file)
    fix_cowan = read_cowan_store(rcg_file)

    gen_rac, gen_cowan = generate_ground_state_rac(l=l, n=n)

    # CF-only: the 10DQ operator uses SHELL_4 blocks which are pure
    # angular factors (no Slater integrals needed).
    operators = ['10DQ']
    xham = [1.0]

    # For fixture: ground manifold is in section 2 (nconf >= 2)
    fix_sec = fix_cowan[2] if len(fix_cowan) > 2 else fix_cowan[0]
    gen_sec = gen_cowan[2]

    matched = 0
    for irrep, butler in [('A1', '0+'), ('A2', '^0+'), ('E', '2+'),
                           ('T1', '1+'), ('T2', '^1+')]:
        fix_evals = _assemble_irrep_hamiltonian(
            fix_rac, fix_sec, butler, operators, xham)
        gen_evals = _assemble_irrep_hamiltonian(
            gen_rac, gen_sec, butler, operators, xham)

        if fix_evals is None:
            assert gen_evals is None, f"Generator has {irrep} but fixture doesn't"
            continue
        assert gen_evals is not None, f"Fixture has {irrep} but generator doesn't"

        assert len(fix_evals) == len(gen_evals), (
            f"{irrep}: dim {len(gen_evals)} != {len(fix_evals)}"
        )

        diff = torch.abs(fix_evals - gen_evals)
        max_diff = diff.max().item()
        assert max_diff < 1e-6, (
            f"{irrep}: max eigenvalue diff = {max_diff:.2e}\n"
            f"  fix: {fix_evals[:5].tolist()}\n"
            f"  gen: {gen_evals[:5].tolist()}"
        )
        matched += 1

    assert matched > 0, "No irreps were compared"


# ---------- Generator smoke tests (no fixture needed) ----------

@pytest.mark.parametrize("l,n", [
    (2, 2), (2, 4), (2, 6), (2, 8),
])
def test_generator_runs_all_even_n(l, n):
    """Generator produces valid output for all even-electron d^N."""
    rac, cowan = generate_ground_state_rac(l=l, n=n)

    # Non-trivial: should have at least one GROUND block with ADD entries
    ground_blocks = [
        b for b in rac.blocks
        if b.kind == 'GROUND' and b.add_entries
    ]
    assert len(ground_blocks) > 0, f"d^{n}: no GROUND blocks with ADD entries"

    # COWAN store section 2 should exist and have tensors
    assert len(cowan) > 2
    assert len(cowan[2]) > 0


# ---------- Empty operator blocks: emitted, not skipped (residual 58) ----------
#
# Residual 44 recorded that the Oh emission path wrote an operator block whether
# or not it had ADD entries while the D4h path skipped an empty one, and left the
# difference open because "dropping them is a byte change". The plan's prescribed
# fix was ``skip_empty=True`` for Oh too -- which is the wrong direction. ttrac's
# ``execute_racer_task`` (ttrac.c:7086-7137) writes the ``REDUCEDMATRIX`` header
# and the matching ``END`` **unconditionally** for every present triad whose
# operator irrep matches the actor; the ADD lines are whatever the inner triad
# loop finds, possibly none. So the empty block is Fortran's own convention, the
# Oh path was faithful, and the D4h skip was the deviation.
#
# The oracle is reference data rather than that source read: the two tests below
# are a pair, the first pinning what the Fortran chain actually emitted and the
# second pinning that the port matches it.

OPERATOR_GROUPS = ('oh', 'td', 'd4h')


def _blocks_by_section(path):
    """``[[(kind, bra, operator, n_add), ...], ...]`` per .rme_rac section.

    A run of ``IRREP`` lines opens a section, which is how ``read_rme_rac_full``
    splits them too; this reader is deliberately independent of it so that the
    fixture evidence does not depend on the parser under test elsewhere.
    """
    import re
    head = re.compile(r'\s*REDUCEDMATRIX\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\d+\s+(\S+)')
    sections, section, block, in_header = [], None, None, False
    for line in open(path):
        if line.strip().startswith('IRREP '):
            if not in_header:
                if block is not None:
                    section.append(block)
                    block = None
                section = []
                sections.append(section)
                in_header = True
            continue
        in_header = False
        m = head.match(line)
        if m:
            if block is not None:
                section.append(block)
            block = [m.group(1), m.group(2), m.group(3), 0]
        elif line.strip().startswith('ADD') and block is not None:
            block[3] += 1
    if block is not None:
        section.append(block)
    return sections


def test_the_fortran_chain_emits_an_operator_block_with_no_add_entries():
    """ttrac writes an empty ``10DQ`` block for a d-shell with no crystal field.

    ``ti4_d0_oh`` is the discriminating fixture: its third section carries real
    ADD entries, and in it the 3d⁰ ground manifold -- an empty d shell, so every
    crystal-field coupling vanishes identically -- gets a ``10DQ`` block with
    zero ADDs sitting immediately after a ``HAMILTONIAN`` block on the same
    irrep that carries one. That rules out the alternative reading, which is
    that empty blocks appear only in the leading placeholder sections.
    """
    path = os.path.join(REF_DIR, 'ti4_d0_oh', 'ti4_d0_oh.rme_rac')
    sections = _blocks_by_section(path)

    mixed = [(i, s) for i, s in enumerate(sections, 1)
             if any(b[3] for b in s) and any(not b[3] for b in s)]
    assert mixed, "no section mixes empty and non-empty blocks"

    # the d^0 ground manifold, in a section that carries data
    found = []
    for i, section in mixed:
        by_key = {(b[0], b[1], b[2]): b[3] for b in section}
        ham = by_key.get(('GROUND', '0+', 'HAMILTONIAN'))
        cf = by_key.get(('GROUND', '0+', '10DQ'))
        if ham and cf == 0:
            found.append(i)
    assert found, (
        "expected a section where GROUND 0+ carries a HAMILTONIAN ADD and an "
        f"empty 10DQ block; got {[(i, sorted(b[:3] + [b[3]] for b in s)) for i, s in mixed]}"
    )


@pytest.mark.parametrize("sym", OPERATOR_GROUPS)
@pytest.mark.parametrize("n", range(1, 10))
def test_every_manifold_irrep_gets_every_operator_block(sym, n):
    """The operator blocks are the full ``irreps x operator_order`` grid.

    No gaps, in any group, at any electron count -- which is the port's side of
    the Fortran convention above. The case this bites is d9: the excited
    manifold is 2p⁵3d¹⁰, a closed d shell, so every crystal-field coupling is
    zero and the blocks are empty. Before residual 58 the D4h path dropped six
    of them there (``10DQ``, ``DT`` and ``DS`` on both ``1/2-`` and ``3/2-``)
    and Oh dropped none, which is the whole of the difference across the
    27-case matrix.
    """
    from multitorch.angular.rac_generator import generate_ledge_template
    from multitorch.angular.symmetry_plan import plan_for_sym

    operators = tuple(plan_for_sym(sym).operator_order)
    rac, _store, _dec = generate_ledge_template(2, n, sym=sym)

    emitted = {(b.bra_sym, b.geometry) for b in rac.blocks
               if b.geometry in operators and b.bra_sym == b.ket_sym}
    expected = {(info.name, op) for info in rac.irreps for op in operators}

    assert emitted == expected, (
        f"{sym} d{n}: missing {sorted(expected - emitted)}, "
        f"unexpected {sorted(emitted - expected)}"
    )


def test_d9_is_where_the_empty_blocks_live_and_they_are_the_only_empty_ones():
    """Pins the measurement: six empty blocks in D4h d9, two in Oh/Td, else none.

    Counted so that a version of the grid test that passed vacuously -- because
    nothing was ever empty -- would fail here instead of going quiet.
    """
    from multitorch.angular.rac_generator import generate_ledge_template
    from multitorch.angular.symmetry_plan import plan_for_sym

    empties = {}
    for sym in OPERATOR_GROUPS:
        operators = tuple(plan_for_sym(sym).operator_order)
        for n in range(1, 10):
            rac, _store, _dec = generate_ledge_template(2, n, sym=sym)
            empty = [(b.bra_sym, b.geometry) for b in rac.blocks
                     if b.geometry in operators and not b.add_entries]
            if empty:
                empties[(sym, n)] = sorted(empty)

    assert set(empties) == {('oh', 9), ('td', 9), ('d4h', 9)}, sorted(empties)
    assert len(empties[('d4h', 9)]) == 6, empties[('d4h', 9)]
    assert len(empties[('oh', 9)]) == 2, empties[('oh', 9)]
    assert len(empties[('td', 9)]) == 2, empties[('td', 9)]
    # all of them are the crystal field on the closed d^10 excited manifold
    assert all(op != 'HAMILTONIAN' for sym_n in empties for _irrep, op in empties[sym_n])


def test_reinstating_the_skip_makes_the_grid_incomplete():
    """The injection: without it the grid test could be vacuous.

    Wraps the one emitter so that it drops an empty block again -- the state the
    D4h path was in before residual 58 -- and requires the grid test to fail.
    """
    from multitorch.angular import rac_generator as rg

    original = rg._emit_operator_blocks

    def skipping(blocks, irrep_infos, operator_order, manifold_irreps,
                 make_adds, kind):
        before = len(blocks)
        original(blocks, irrep_infos, operator_order, manifold_irreps,
                 make_adds, kind)
        blocks[before:] = [b for b in blocks[before:] if b.add_entries]

    rg._emit_operator_blocks = skipping
    try:
        with pytest.raises(AssertionError, match="missing"):
            test_every_manifold_irrep_gets_every_operator_block('d4h', 9)
    finally:
        rg._emit_operator_blocks = original

    # and with the real emitter back it passes again
    test_every_manifold_irrep_gets_every_operator_block('d4h', 9)
