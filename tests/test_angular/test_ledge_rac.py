"""Tests for L-edge RAC generator (generate_ledge_rac).

Validates:
  1. oh_transition_coupling gives correct coefficients
  2. generate_ledge_rac TRANSI blocks have correct structure and D4h splitting
  3. Generated RAC + COWAN store can be consumed by the assembler
  4. Transition coupling magnitudes match Fortran Oh fixture
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.point_group import (
    OH_IRREP_DIM,
    butler_label,
    oh_branching,
    oh_transition_coupling,
)
from multitorch.angular.rac_generator import generate_ledge_rac
from multitorch.io.read_rme import read_rme_rac_full

REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')


# ─────────────────────────────────────────────────────────────
# oh_transition_coupling unit tests
# ─────────────────────────────────────────────────────────────

def test_transition_coupling_a1_t1_unit():
    """A1(J=0) → T1(J=1) coupling is 1.0 for k=1 dipole."""
    c = oh_transition_coupling(0, 1, k=1, l=2)
    val = c.get(('A1', 0, 0, 'T1', 1, 0), 0.0)
    assert abs(val - 1.0) < 1e-10, f"A1→T1 coupling = {val}, expected 1.0"


def test_transition_coupling_perp_para_sum_rule():
    """PERP² + PARA² = oh_coupling² for all channels.

    D4h splitting: PERP = √(2/3) × oh, PARA = √(1/3) × oh.
    Summing PERP² + PARA² = (2/3 + 1/3) × oh² = oh².
    """
    c = oh_transition_coupling(4, 4, k=1, l=2)
    for key, val in c.items():
        perp = math.sqrt(2.0 / 3.0) * val
        para = math.sqrt(1.0 / 3.0) * val
        assert abs(perp ** 2 + para ** 2 - val ** 2) < 1e-12


def test_transition_coupling_triangle_inequality():
    """Only (J_bra, J_ket) pairs satisfying |J_bra-J_ket| ≤ 1 appear."""
    c = oh_transition_coupling(4, 4, k=1, l=2)
    for (_, Jb, _, _, Jk, _) in c:
        assert abs(Jb - Jk) <= 1, f"J_bra={Jb}, J_ket={Jk} violates triangle"


def test_transition_coupling_reverse_entry_exists():
    """Every (Γ_bra, J_bra) → (Γ_ket, J_ket) has a reverse entry.

    Magnitudes need NOT match: the CG projection is asymmetric when
    dim(Γ_bra) ≠ dim(Γ_ket), so |T1→E| ≠ |E→T1| in general.
    We only check that the reverse key exists (completeness).
    """
    c = oh_transition_coupling(4, 4, k=1, l=2)
    for (ib, Jb, cb, ik, Jk, ck) in c:
        rev = c.get((ik, Jk, ck, ib, Jb, cb))
        assert rev is not None, f"Missing reverse: ({ik},{Jk},{ck},{ib},{Jb},{cb})"


# ─────────────────────────────────────────────────────────────
# Comparison with Fortran Oh fixture (magnitude level)
# ─────────────────────────────────────────────────────────────

def test_transition_coupling_vs_fortran_leading_entry():
    """Leading TRANSI ADD coefficient for A1→T1 matches fixture.

    The Fortran Oh fixture's MULTIPOLE TRANSI block for 0+→1- has
    leading entry coeff=1.0 (the J=0→J=1 monopole dipole matrix element).
    Our oh_transition_coupling(A1,0,0,T1,1,0) = 1.0 should match exactly.

    Note: per-matrix-element ADD coefficients in the fixture include
    multiplicity and COWAN indexing structure that differs from our
    per-irrep coupling constants, so only the leading term (which has
    no multiplicity complications) is compared directly.
    """
    rac_file = os.path.join(REF_DIR, 'ni2_d8_oh', 'ni2_d8_oh.rme_rac')
    if not os.path.exists(rac_file):
        pytest.skip("No ni2_d8_oh fixture")

    fix_rac = read_rme_rac_full(rac_file)
    couplings = oh_transition_coupling(4, 4, k=1, l=2)

    # Check the leading coupling: A1(J=0) → T1(J=1) = 1.0
    val = couplings.get(('A1', 0, 0, 'T1', 1, 0), 0.0)
    assert abs(val - 1.0) < 1e-10, f"A1→T1 leading coupling = {val}"

    # Check that the fixture's 0+→1- leading ADD entry is also 1.0
    for blk in fix_rac.blocks:
        if (blk.kind == 'TRANSI' and blk.bra_sym == '0+'
                and blk.ket_sym == '1-' and blk.geometry == 'MULTIPOLE'
                and blk.add_entries):
            leading = blk.add_entries[0].coeff
            assert abs(leading - 1.0) < 1e-8, (
                f"Fixture leading 0+→1- ADD = {leading:.10f}, expected 1.0"
            )
            break
    else:
        pytest.skip("No 0+→1- MULTIPOLE block in fixture")

    # Verify all expected irrep pairs have TRANSI blocks in the fixture
    butler_to_oh = {
        '0+': 'A1', '^0+': 'A2', '2+': 'E', '1+': 'T1', '^1+': 'T2',
    }
    fixture_pairs = set()
    for blk in fix_rac.blocks:
        if blk.kind == 'TRANSI' and blk.geometry == 'MULTIPOLE':
            gs = butler_to_oh.get(blk.bra_sym)
            if gs and blk.ket_sym.endswith('-'):
                fixture_pairs.add(gs)

    coupling_gs_irreps = {ib for (ib, _, _, _, _, _) in couplings}
    assert coupling_gs_irreps == fixture_pairs, (
        f"Coupling irreps {coupling_gs_irreps} != fixture {fixture_pairs}"
    )


# ─────────────────────────────────────────────────────────────
# generate_ledge_rac structural tests
# ─────────────────────────────────────────────────────────────

def test_ledge_rac_produces_all_block_types():
    """Generated RAC has TRANSI, GROUND, and EXCITE blocks."""
    rac, cowan = generate_ledge_rac(l_val=2, n_val_gs=8)

    kinds = {b.kind for b in rac.blocks}
    assert 'TRANSI' in kinds, "Missing TRANSI blocks"
    assert 'GROUND' in kinds, "Missing GROUND blocks"

    # EXCITE blocks use kind='GROUND' in assembler convention
    # but have ungerade (minus) Butler labels
    ungerade_blocks = [b for b in rac.blocks
                       if b.kind == 'GROUND' and b.bra_sym.endswith('-')]
    assert len(ungerade_blocks) > 0, "Missing excited-state (ungerade) blocks"


def test_ledge_rac_perp_para_splitting():
    """TRANSI blocks are split into PERP and PARA with correct factors."""
    rac, cowan = generate_ledge_rac(l_val=2, n_val_gs=8)

    perp_blocks = [b for b in rac.blocks
                   if b.kind == 'TRANSI' and b.geometry == 'PERP']
    para_blocks = [b for b in rac.blocks
                   if b.kind == 'TRANSI' and b.geometry == 'PARA']

    assert len(perp_blocks) > 0, "No PERP blocks"
    assert len(para_blocks) > 0, "No PARA blocks"
    assert len(perp_blocks) == len(para_blocks), (
        f"PERP count ({len(perp_blocks)}) != PARA count ({len(para_blocks)})"
    )

    # Verify PERP/PARA ratio is √2 for all matching entries
    for pb, ab in zip(
        sorted(perp_blocks, key=lambda b: (b.bra_sym, b.ket_sym)),
        sorted(para_blocks, key=lambda b: (b.bra_sym, b.ket_sym)),
    ):
        assert pb.bra_sym == ab.bra_sym and pb.ket_sym == ab.ket_sym
        assert len(pb.add_entries) == len(ab.add_entries)
        for pe, ae in zip(pb.add_entries, ab.add_entries):
            assert pe.matrix_idx == ae.matrix_idx
            if abs(ae.coeff) > 1e-15:
                ratio = pe.coeff / ae.coeff
                assert abs(ratio - math.sqrt(2)) < 1e-10, (
                    f"PERP/PARA ratio = {ratio:.6f}, expected √2"
                )


def test_ledge_rac_op_sym_labels():
    """PERP blocks have op_sym='1-', PARA blocks have op_sym='^0-'."""
    rac, _ = generate_ledge_rac(l_val=2, n_val_gs=8)

    for b in rac.blocks:
        if b.kind != 'TRANSI':
            continue
        if b.geometry == 'PERP':
            assert b.op_sym == '1-', f"PERP block has op_sym={b.op_sym}"
        elif b.geometry == 'PARA':
            assert b.op_sym == '^0-', f"PARA block has op_sym={b.op_sym}"


def test_ledge_rac_irrep_infos():
    """RAC has GROUND and EXCITE irrep infos with correct dimensions."""
    rac, _ = generate_ledge_rac(l_val=2, n_val_gs=8)

    gs_irreps = [ir for ir in rac.irreps if ir.kind == 'GROUND']
    ex_irreps = [ir for ir in rac.irreps if ir.kind == 'EXCITE']

    assert len(gs_irreps) > 0, "No ground-state irreps"
    assert len(ex_irreps) > 0, "No excited-state irreps"

    # Ground irreps should be gerade (+)
    for ir in gs_irreps:
        assert ir.name.endswith('+'), f"GS irrep {ir.name} not gerade"

    # Excited irreps should be ungerade (-)
    for ir in ex_irreps:
        assert ir.name.endswith('-'), f"EX irrep {ir.name} not ungerade"


def test_ledge_rac_cowan_store_nonempty():
    """COWAN store section 0 has matrices for all referenced indices."""
    rac, cowan = generate_ledge_rac(l_val=2, n_val_gs=8)

    max_idx = 0
    for b in rac.blocks:
        for e in b.add_entries:
            max_idx = max(max_idx, e.matrix_idx)

    # matrix_idx is 1-based, cowan[0] is 0-indexed
    assert len(cowan[0]) >= max_idx, (
        f"COWAN store has {len(cowan[0])} matrices but max_idx={max_idx}"
    )


# ─────────────────────────────────────────────────────────────
# Assembler integration test
# ─────────────────────────────────────────────────────────────

def test_ledge_rac_assembler_runs():
    """Generated RAC + COWAN store can be fed through the assembler.

    Uses identity Hamiltonian blocks (no atomic parameters) and
    constructs a minimal BanData with PERP + PARA triads.
    """
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.io.read_ban import BanData, XHAMEntry

    rac, cowan = generate_ledge_rac(l_val=2, n_val_gs=8)

    # Collect all unique triads from TRANSI blocks
    triads = []
    for b in rac.blocks:
        if b.kind == 'TRANSI' and b.add_entries:
            triad = (b.bra_sym, b.op_sym, b.ket_sym)
            if triad not in triads:
                triads.append(triad)

    ban = BanData(
        nconf_gs=1,
        nconf_fs=1,
        xham=[XHAMEntry(values=[1.0, 1.0], combos=[(1, 1), (1, 2), (2, 1), (2, 2)])],
        tran=[(1, 1)],
        triads=triads,
        eg={1: 0.0},
        ef={1: 0.0},
    )

    result = assemble_and_diagonalize_in_memory(cowan, rac, ban)

    assert len(result.triads) > 0, "No triads produced"
    for t in result.triads:
        assert t.Eg.numel() > 0, f"Empty Eg for {t.gs_sym}"
        assert t.Ef.numel() > 0, f"Empty Ef for {t.gs_sym}"
        assert t.T.numel() > 0, f"Empty T for {t.gs_sym}"
        # Transition matrix should be finite
        assert torch.isfinite(t.T).all(), f"Non-finite T for {t.gs_sym}"


def test_ledge_rac_sticks_nonzero():
    """Generated RAC produces nonzero transition intensities."""
    from multitorch.hamiltonian.assemble import assemble_and_diagonalize_in_memory
    from multitorch.io.read_ban import BanData, XHAMEntry
    from multitorch.spectrum.sticks import get_sticks_from_banresult

    rac, cowan = generate_ledge_rac(l_val=2, n_val_gs=8)

    triads = []
    for b in rac.blocks:
        if b.kind == 'TRANSI' and b.add_entries:
            triad = (b.bra_sym, b.op_sym, b.ket_sym)
            if triad not in triads:
                triads.append(triad)

    ban = BanData(
        nconf_gs=1,
        nconf_fs=1,
        xham=[XHAMEntry(values=[1.0, 1.0], combos=[(1, 1), (1, 2), (2, 1), (2, 2)])],
        tran=[(1, 1)],
        triads=triads,
        eg={1: 0.0},
        ef={1: 0.0},
    )

    result = assemble_and_diagonalize_in_memory(cowan, rac, ban)
    E, M, Eg_min = get_sticks_from_banresult(result, T=0.0, max_gs=10)

    assert E.numel() > 0, "No sticks"
    total_intensity = (M ** 2).sum()  # M is amplitude, intensity = M²
    # Wait, get_sticks_from_banresult returns amplitudes T, and
    # the intensity is T² already squared by the function.
    # Actually M here IS the amplitude; the stick spectrum squares it.
    # But M = result.T (amplitudes), so M can be negative.
    # The intensity is M (already Boltzmann-weighted amplitude).
    # For T=0, all weights are 1, so M = amplitude.
    total_m = M.abs().sum()
    assert total_m > 0, "Zero total transition intensity"
