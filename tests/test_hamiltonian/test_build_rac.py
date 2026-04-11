"""
Track C3d tests for ``build_rac_in_memory`` and the COWAN section plan.

What this validates
-------------------
1. The loader-based ``build_rac_in_memory`` returns a ``RACFileFull``
   that's byte-equivalent to the direct ``read_rme_rac_full`` parser.
2. The derived ``SectionPlan`` has the same per-section matrix counts
   as the parsed ``read_cowan_store`` output (the contract that C3e
   must satisfy).
3. The ``classify_block_section`` rule routes the standard nid8ct
   blocks to the expected sections (TRANSI conf-1 → 0, TRANSI conf-2
   → 1, GROUND/EXCITE/HYBR ``+`` parity → 2, ``-`` parity → 3).
4. Every ADD entry in the parsed RAC points to a valid slot in the
   plan, and the slot's row capacity is at least ``add.nbra``.
5. ``validate_section_plan`` raises ``ValueError`` on a deliberately
   damaged ADD entry.
6. The recovered shapes are *consistent* with the parsed COWAN store
   (column count exact, row count ≥ max ``add.nbra``).
7. End-to-end round trip: feeding the plan + parsed COWAN sections back
   into ``assemble_matrix_from_adds`` reproduces the same Hamiltonian
   blocks the file-based ``assemble_and_diagonalize`` produces. This is
   the C3f-style parity contract reduced to the C3d boundary.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from multitorch.hamiltonian.assemble import (
    assemble_and_diagonalize,
    assemble_and_diagonalize_in_memory,
)
from multitorch.hamiltonian.build_rac import (
    SectionEntry,
    SectionPlan,
    build_rac_in_memory,
    classify_block_section,
    derive_section_plan,
    validate_section_plan,
)
from multitorch.io.read_ban import read_ban
from multitorch.io.read_rme import (
    RACFileFull,
    read_cowan_store,
    read_rme_rac_full,
)

REFDATA = Path(__file__).parent.parent / "reference_data"
NID8CT = REFDATA / "nid8ct"
NID8CT_BAN = NID8CT / "nid8ct.ban"
NID8CT_RAC = NID8CT / "nid8ct.rme_rac"
NID8CT_RCG = NID8CT / "nid8ct.rme_rcg"

# Expected COWAN section sizes for the nid8ct fixture, hand-extracted
# from `read_cowan_store(NID8CT_RCG)`. These are the contract numbers
# C3e must reproduce when it builds the COWAN store from physical
# parameters.
NID8CT_EXPECTED_SECTION_SIZES = [22, 24, 167, 142]


# ─────────────────────────────────────────────────────────────
# Module-level fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nid8ct_ban():
    return read_ban(NID8CT_BAN)


@pytest.fixture(scope="module")
def nid8ct_rac_parsed():
    return read_rme_rac_full(NID8CT_RAC)


@pytest.fixture(scope="module")
def nid8ct_cowan_parsed():
    return read_cowan_store(NID8CT_RCG)


@pytest.fixture(scope="module")
def nid8ct_built(nid8ct_ban):
    return build_rac_in_memory(
        nid8ct_ban,
        source_rac_path=NID8CT_RAC,
        source_rcg_path=NID8CT_RCG,
    )


# ─────────────────────────────────────────────────────────────
# Return type and shape
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_returns_rac_and_plan(nid8ct_built):
    rac, plan = nid8ct_built
    assert isinstance(rac, RACFileFull)
    assert isinstance(plan, SectionPlan)


@pytest.mark.phase4
def test_built_rac_block_count_matches_parser(nid8ct_built, nid8ct_rac_parsed):
    rac, _ = nid8ct_built
    assert len(rac.blocks) == len(nid8ct_rac_parsed.blocks)
    assert len(rac.irreps) == len(nid8ct_rac_parsed.irreps)


@pytest.mark.phase4
def test_built_rac_byte_equivalent(nid8ct_built, nid8ct_rac_parsed):
    """Block-by-block (kind, syms, geometry, dims, ADD list) equivalence."""
    rac, _ = nid8ct_built
    for b1, b2 in zip(rac.blocks, nid8ct_rac_parsed.blocks):
        assert b1.kind == b2.kind
        assert b1.bra_sym == b2.bra_sym
        assert b1.op_sym == b2.op_sym
        assert b1.ket_sym == b2.ket_sym
        assert b1.geometry == b2.geometry
        assert b1.n_bra == b2.n_bra
        assert b1.n_ket == b2.n_ket
        assert len(b1.add_entries) == len(b2.add_entries)
        for a1, a2 in zip(b1.add_entries, b2.add_entries):
            assert a1.matrix_idx == a2.matrix_idx
            assert a1.bra == a2.bra
            assert a1.ket == a2.ket
            assert a1.nbra == a2.nbra
            assert a1.nket == a2.nket
            assert a1.coeff == a2.coeff


# ─────────────────────────────────────────────────────────────
# SectionPlan structure
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_plan_has_four_sections(nid8ct_built):
    _, plan = nid8ct_built
    assert plan.n_sections == 4


@pytest.mark.phase4
def test_plan_section_sizes_match_parsed_cowan(nid8ct_built, nid8ct_cowan_parsed):
    _, plan = nid8ct_built
    sizes = [plan.section_size(i) for i in range(plan.n_sections)]
    expected = [len(s) for s in nid8ct_cowan_parsed]
    assert sizes == expected
    assert sizes == NID8CT_EXPECTED_SECTION_SIZES


@pytest.mark.phase4
def test_plan_total_matches_cowan_total(nid8ct_built, nid8ct_cowan_parsed):
    _, plan = nid8ct_built
    assert plan.total_matrices() == sum(len(s) for s in nid8ct_cowan_parsed)


@pytest.mark.phase4
def test_plan_entry_columns_match_parsed_shapes(nid8ct_built, nid8ct_cowan_parsed):
    """Required column count must equal the parsed source matrix's column count."""
    _, plan = nid8ct_built
    for s, sec_mats in enumerate(nid8ct_cowan_parsed):
        for j, mat in enumerate(sec_mats):
            entry = plan.get(s, j + 1)
            assert entry.required_n_cols == int(mat.shape[1])
            # Row count must not exceed parsed shape (it's the max nbra)
            assert entry.required_n_rows <= int(mat.shape[0])


@pytest.mark.phase4
def test_plan_entries_have_correct_indices(nid8ct_built):
    _, plan = nid8ct_built
    for s in range(plan.n_sections):
        for j, entry in enumerate(plan.sections[s]):
            assert entry.section_idx == s
            assert entry.matrix_idx == j + 1


# ─────────────────────────────────────────────────────────────
# classify_block_section dispatch
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
@pytest.mark.parametrize(
    "sym,expected", [
        ("0+", True), ("^0+", True), ("1+", True), ("2+", True),
        ("0-", False), ("^0-", False), ("1-", False), ("^2-", False),
    ],
)
def test_parity_classification(sym, expected):
    from multitorch.hamiltonian.build_rac import _is_plus_parity
    assert _is_plus_parity(sym) is expected


@pytest.mark.phase4
def test_classify_ground_plus_goes_to_section_2(nid8ct_rac_parsed):
    blk = next(
        b for b in nid8ct_rac_parsed.blocks
        if b.kind == "GROUND" and b.geometry == "HAMILTONIAN" and b.bra_sym == "0+"
    )
    assert classify_block_section(blk, nconf=2) == 2


@pytest.mark.phase4
def test_classify_excite_minus_goes_to_section_3(nid8ct_rac_parsed):
    blk = next(
        b for b in nid8ct_rac_parsed.blocks
        if b.kind == "EXCITE" and b.geometry == "HAMILTONIAN" and b.bra_sym == "1-"
    )
    assert classify_block_section(blk, nconf=2) == 3


@pytest.mark.phase4
def test_classify_hybr_plus_goes_to_section_2(nid8ct_rac_parsed):
    blk = next(
        b for b in nid8ct_rac_parsed.blocks
        if b.kind == "TRANSI" and "HYBR" in b.geometry and b.bra_sym.endswith("+")
    )
    assert classify_block_section(blk, nconf=2) == 2


@pytest.mark.phase4
def test_classify_hybr_minus_goes_to_section_3(nid8ct_rac_parsed):
    blk = next(
        b for b in nid8ct_rac_parsed.blocks
        if b.kind == "TRANSI" and "HYBR" in b.geometry and b.bra_sym.endswith("-")
    )
    assert classify_block_section(blk, nconf=2) == 3


@pytest.mark.phase4
def test_nconf_one_collapses_to_section_zero(nid8ct_rac_parsed):
    """For single-config fixtures every block must land in section 0."""
    for blk in nid8ct_rac_parsed.blocks[:20]:
        assert classify_block_section(blk, nconf=1) == 0


# ─────────────────────────────────────────────────────────────
# TRANSI conf-1 / conf-2 split
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_transi_conf_split_uses_sections_0_and_1(nid8ct_built, nid8ct_rac_parsed):
    """Both sections 0 and 1 must contain TRANSI references."""
    _, plan = nid8ct_built
    # Section 0 and 1 should each have at least one referenced entry.
    sec0_refs = sum(1 for e in plan.sections[0] if e.n_references > 0)
    sec1_refs = sum(1 for e in plan.sections[1] if e.n_references > 0)
    assert sec0_refs > 0
    assert sec1_refs > 0


# ─────────────────────────────────────────────────────────────
# validate_section_plan
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_validate_passes_on_clean_fixture(nid8ct_built, nid8ct_ban):
    rac, plan = nid8ct_built
    # Should not raise.
    validate_section_plan(plan, rac, nconf=nid8ct_ban.nconf_gs)


@pytest.mark.phase4
def test_validate_raises_on_out_of_range_matrix_idx(nid8ct_built, nid8ct_ban):
    rac, plan = nid8ct_built
    bad_rac = copy.deepcopy(rac)
    # Pick a TRANSI dipole block (section 0) and corrupt one ADD entry.
    target = next(
        b for b in bad_rac.blocks
        if b.kind == "TRANSI" and "HYBR" not in b.geometry and b.add_entries
    )
    target.add_entries[0].matrix_idx = 9999
    with pytest.raises(ValueError, match="out of range"):
        validate_section_plan(plan, bad_rac, nconf=nid8ct_ban.nconf_gs)


@pytest.mark.phase4
def test_validate_raises_on_oversized_nbra(nid8ct_built, nid8ct_ban):
    rac, plan = nid8ct_built
    bad_rac = copy.deepcopy(rac)
    target = next(
        b for b in bad_rac.blocks
        if b.kind == "GROUND" and b.geometry == "HAMILTONIAN" and b.add_entries
    )
    target.add_entries[0].nbra = 999
    with pytest.raises(ValueError, match="exceeds slot capacity"):
        validate_section_plan(plan, bad_rac, nconf=nid8ct_ban.nconf_gs)


# ─────────────────────────────────────────────────────────────
# derive_section_plan can be called standalone
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_derive_section_plan_standalone(nid8ct_rac_parsed, nid8ct_cowan_parsed):
    plan = derive_section_plan(nid8ct_rac_parsed, nid8ct_cowan_parsed, nconf=2)
    assert plan.n_sections == 4
    assert [plan.section_size(i) for i in range(4)] == NID8CT_EXPECTED_SECTION_SIZES


@pytest.mark.phase4
def test_n_references_sum_equals_total_add_entries(
    nid8ct_built, nid8ct_rac_parsed,
):
    """Every ADD entry contributes exactly one reference to its slot."""
    _, plan = nid8ct_built
    total_refs = sum(
        e.n_references for sec in plan.sections for e in sec
    )
    total_adds = sum(len(b.add_entries) for b in nid8ct_rac_parsed.blocks)
    assert total_refs == total_adds


# ─────────────────────────────────────────────────────────────
# End-to-end: feed (parsed COWAN, built RAC) into the assembler
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_in_memory_builder_round_trips_through_assembler(
    nid8ct_ban, nid8ct_built, nid8ct_cowan_parsed,
):
    """Parsed COWAN + built RAC + parsed BAN must equal full file path.

    This is the C3d → assembler boundary contract: a SectionPlan-derived
    pipeline must reproduce ``BanResult`` byte-for-byte against the
    file-based wrapper. C3e will swap ``nid8ct_cowan_parsed`` for a
    physically-built COWAN store; until then, this test guarantees that
    the C3d half of the in-memory pipeline is sound.
    """
    rac, _ = nid8ct_built
    built = assemble_and_diagonalize_in_memory(nid8ct_cowan_parsed, rac, nid8ct_ban)
    from_disk = assemble_and_diagonalize(NID8CT_RCG, NID8CT_RAC, NID8CT_BAN)

    assert len(built.triads) == len(from_disk.triads)
    for tb, td in zip(built.triads, from_disk.triads):
        import torch
        assert torch.allclose(tb.Eg, td.Eg, atol=1e-12)
        assert torch.allclose(tb.Ef, td.Ef, atol=1e-12)
        assert torch.allclose(tb.T, td.T, atol=1e-12)
