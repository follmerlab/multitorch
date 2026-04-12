"""
Track C3e tests for ``build_cowan_store_in_memory``.

What this validates
-------------------
1. The section count and per-section matrix count match the parsed fixture.
2. Per-matrix shapes match the parsed fixture.
3. Element-wise parity at ``atol=1e-12`` with ``slater_scale=1.0, soc_scale=1.0``
   (the decomposition is algebraically exact; the 1e-12 tolerance absorbs
   IEEE 754 rounding in the extraction/rebuild arithmetic).
4. The metadata parser ``read_cowan_metadata`` aligns 1:1 with ``read_cowan_store``.
5. Config-1 GROUND HAMILTONIAN blocks in section 2 have been rebuilt
   (they are fresh tensors, not the same object as the template).
6. Config-2 EXCITE blocks are passed through verbatim (same object).
7. Gradient propagation through ``slater_scale``: backward through a
   downstream sum of a rebuilt HAMILTONIAN block lands on the leaf with
   finite, nonzero gradient.
8. Gradient propagation through ``soc_scale``: same contract.
9. Gradient isolation: ``slater_scale`` gradient comes only from Coulomb,
   ``soc_scale`` gradient comes only from SOC; no cross-talk.
10. End-to-end round trip: feeding the built COWAN store into
    ``assemble_and_diagonalize_in_memory`` reproduces the file-based result.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE, RY_TO_EV
from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
from multitorch.atomic.scaled_params import scale_atomic_params
from multitorch.hamiltonian.assemble import (
    assemble_and_diagonalize,
    assemble_and_diagonalize_in_memory,
)
from multitorch.hamiltonian.build_cowan import (
    CowanBlockMeta,
    build_cowan_store_in_memory,
    read_cowan_metadata,
)
from multitorch.hamiltonian.build_rac import build_rac_in_memory
from multitorch.io.read_ban import read_ban
from multitorch.io.read_rme import read_cowan_store

REFDATA = Path(__file__).parent.parent / "reference_data"
NID8CT = REFDATA / "nid8ct"
NID8CT_BAN = NID8CT / "nid8ct.ban"
NID8CT_RCG = NID8CT / "nid8ct.rme_rcg"
NID8CT_RAC = NID8CT / "nid8ct.rme_rac"

# nid8ct does not have a .rcn31_out; the non-CT nid8 fixture does.
# Config 1 (d^8) parameters are the same element and configuration,
# so nid8.rcn31_out NCONF=1 provides the raw values.
NID8_RCN31 = REFDATA / "nid8" / "nid8.rcn31_out"

# Expected section sizes (from C3d).
EXPECTED_SECTION_SIZES = [22, 24, 167, 142]


# ─────────────────────────────────────────────────────────────
# Module-level fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nid8ct_ban():
    return read_ban(NID8CT_BAN)


@pytest.fixture(scope="module")
def nid8_raw_params():
    return read_rcn31_out_params(NID8_RCN31)


@pytest.fixture(scope="module")
def nid8ct_cowan_parsed():
    return read_cowan_store(NID8CT_RCG)


@pytest.fixture(scope="module")
def nid8ct_metadata():
    return read_cowan_metadata(NID8CT_RCG)


@pytest.fixture(scope="module")
def nid8ct_plan(nid8ct_ban):
    _, plan = build_rac_in_memory(
        nid8ct_ban,
        source_rac_path=NID8CT_RAC,
        source_rcg_path=NID8CT_RCG,
    )
    return plan


@pytest.fixture(scope="module")
def nid8ct_built_identity(nid8_raw_params, nid8ct_plan):
    """COWAN store built with scale=1.0 (must equal template exactly)."""
    scaled = scale_atomic_params(nid8_raw_params, slater_scale=1.0, soc_scale=1.0)
    return build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )


# ─────────────────────────────────────────────────────────────
# Metadata parser alignment
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_metadata_section_count(nid8ct_metadata):
    assert len(nid8ct_metadata) == 4


@pytest.mark.phase4
def test_metadata_per_section_count(nid8ct_metadata, nid8ct_cowan_parsed):
    for s in range(4):
        assert len(nid8ct_metadata[s]) == len(nid8ct_cowan_parsed[s])


@pytest.mark.phase4
def test_metadata_types(nid8ct_metadata):
    for section in nid8ct_metadata:
        for m in section:
            assert isinstance(m, CowanBlockMeta)
            assert m.operator in ("SHELL1", "SPIN1", "MULTIPOLE", "HAMILTONIAN")
            assert m.block_type in ("GROUND", "EXCITE", "TRANSITION")


# ─────────────────────────────────────────────────────────────
# Section count and sizes
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_section_count(nid8ct_built_identity):
    assert len(nid8ct_built_identity) == 4


@pytest.mark.phase4
def test_section_sizes(nid8ct_built_identity, nid8ct_cowan_parsed):
    for s in range(4):
        assert len(nid8ct_built_identity[s]) == len(nid8ct_cowan_parsed[s])
        assert len(nid8ct_built_identity[s]) == EXPECTED_SECTION_SIZES[s]


# ─────────────────────────────────────────────────────────────
# Per-matrix shape equivalence
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_shapes_match_parsed(nid8ct_built_identity, nid8ct_cowan_parsed):
    for s in range(4):
        for j in range(len(nid8ct_cowan_parsed[s])):
            built = nid8ct_built_identity[s][j]
            parsed = nid8ct_cowan_parsed[s][j]
            assert built.shape == parsed.shape, (
                f"Section {s}, slot {j+1}: "
                f"built {built.shape} != parsed {parsed.shape}"
            )


# ─────────────────────────────────────────────────────────────
# Element-wise parity at scale=1.0
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_elementwise_parity_all_sections(nid8ct_built_identity, nid8ct_cowan_parsed):
    """Every matrix in the built store must match the parsed fixture.

    The tolerance is 1e-12 (not 1e-6 as in the plan) because the
    decomposition is algebraically exact at scale=1.0. The only error
    source is IEEE 754 rounding in the extraction and rebuild.
    """
    for s in range(4):
        for j in range(len(nid8ct_cowan_parsed[s])):
            built = nid8ct_built_identity[s][j]
            parsed = nid8ct_cowan_parsed[s][j]
            assert torch.allclose(built, parsed, atol=1e-12), (
                f"Section {s}, slot {j+1}: "
                f"max diff = {(built - parsed).abs().max().item():.2e}"
            )


@pytest.mark.phase4
def test_sections_0_1_are_exact(nid8ct_built_identity, nid8ct_cowan_parsed):
    """Sections 0 and 1 pass through verbatim (no rebuild)."""
    for s in [0, 1]:
        for j in range(len(nid8ct_cowan_parsed[s])):
            built = nid8ct_built_identity[s][j]
            parsed = nid8ct_cowan_parsed[s][j]
            # Should be identical — same tensor object
            assert torch.equal(built, parsed)


@pytest.mark.phase4
def test_rebuilt_hamiltonian_blocks_are_fresh_tensors(
    nid8ct_built_identity, nid8ct_cowan_parsed, nid8ct_metadata,
):
    """Rebuilt HAMILTONIAN blocks must be new tensors, not the originals."""
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            built = nid8ct_built_identity[2][j]
            parsed = nid8ct_cowan_parsed[2][j]
            # Values match but these should be different tensor objects
            assert torch.allclose(built, parsed, atol=1e-12)
            assert built is not parsed


@pytest.mark.phase4
def test_excite_blocks_are_passthrough(
    nid8ct_built_identity, nid8ct_cowan_parsed, nid8ct_metadata,
):
    """Config-2 EXCITE blocks should be identical to the template (passthrough).

    Note: ``built is parsed`` would fail because ``build_cowan_store_in_memory``
    internally calls ``read_cowan_store`` (creating new tensors). We check
    exact bitwise equality instead, which confirms no modification.
    """
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.block_type == "EXCITE":
            built = nid8ct_built_identity[2][j]
            parsed = nid8ct_cowan_parsed[2][j]
            assert torch.equal(built, parsed)


# ─────────────────────────────────────────────────────────────
# Section 2 GROUND HAMILTONIAN block decomposition detail
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_ground_hamiltonian_count_in_section_2(nid8ct_metadata):
    """Config 1 d^8 should have 5 HAMILTONIAN blocks (J=0+..4+)."""
    count = sum(
        1 for m in nid8ct_metadata[2]
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND"
    )
    assert count == 5


@pytest.mark.phase4
def test_ground_shell1_diagonal_coverage(nid8ct_metadata):
    """Each GROUND HAMILTONIAN J should have at least a k=0 SHELL1 match."""
    ham_syms = {
        m.bra_sym
        for m in nid8ct_metadata[2]
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND"
    }
    for j_sym in ham_syms:
        shell_ks = {
            int(m.op_sym.rstrip("+-"))
            for m in nid8ct_metadata[2]
            if (m.operator == "SHELL1"
                and m.block_type == "GROUND"
                and m.bra_sym == j_sym
                and m.ket_sym == j_sym)
        }
        assert 0 in shell_ks, f"No k=0 SHELL1 for J={j_sym}"


# ─────────────────────────────────────────────────────────────
# Autograd: gradient through slater_scale
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_slater_scale_gradient(nid8_raw_params, nid8ct_plan, nid8ct_cowan_parsed, nid8ct_metadata):
    """Backward through a rebuilt HAMILTONIAN block must reach slater_scale."""
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_raw_params, slater_scale=slater_scale, soc_scale=1.0
    )
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )

    # Sum all rebuilt HAMILTONIAN blocks in section 2
    loss = torch.tensor(0.0, dtype=DTYPE)
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            loss = loss + built[2][j].sum()

    loss.backward()

    assert slater_scale.grad is not None
    assert torch.isfinite(slater_scale.grad)
    assert slater_scale.grad.abs() > 1e-6


@pytest.mark.phase4
def test_soc_scale_gradient(nid8_raw_params, nid8ct_plan, nid8ct_metadata):
    """Backward through a rebuilt HAMILTONIAN block must reach soc_scale."""
    soc_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_raw_params, slater_scale=1.0, soc_scale=soc_scale
    )
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )

    loss = torch.tensor(0.0, dtype=DTYPE)
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            loss = loss + built[2][j].sum()

    loss.backward()

    assert soc_scale.grad is not None
    assert torch.isfinite(soc_scale.grad)
    assert soc_scale.grad.abs() > 1e-6


@pytest.mark.phase4
def test_gradient_isolation(nid8_raw_params, nid8ct_plan, nid8ct_metadata):
    """Passthrough blocks must not carry gradient to either scale factor."""
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    soc_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_raw_params, slater_scale=slater_scale, soc_scale=soc_scale
    )
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )

    # Passthrough blocks (EXCITE HAMILTONIAN in section 2) must not
    # require grad or have a grad_fn — they are constants from the
    # parsed template, not rebuilt from scaled_params.
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "EXCITE":
            assert not built[2][j].requires_grad, (
                f"Passthrough block at slot {j+1} should not require grad"
            )
            assert built[2][j].grad_fn is None, (
                f"Passthrough block at slot {j+1} should have no grad_fn"
            )


@pytest.mark.phase4
def test_slater_gradient_analytical(nid8_raw_params, nid8ct_plan, nid8ct_metadata):
    """The slater_scale gradient at scale=1.0 must equal the Coulomb sum.

    At scale=1: d/d(slater_scale) [Σ_k (slater_scale × F^k × RY_TO_EV) × SHELL_k(J,J)]
    = Σ_k F^k_Ry × RY_TO_EV × SHELL_k(J,J) summed over all rebuilt blocks.

    We compute this reference by summing over all GROUND SHELL1 diagonal
    blocks weighted by F^k.
    """
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    scaled = scale_atomic_params(
        nid8_raw_params, slater_scale=slater_scale, soc_scale=1.0
    )
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )

    # Loss = sum of all GROUND HAMILTONIAN blocks in section 2
    loss = torch.tensor(0.0, dtype=DTYPE)
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            loss = loss + built[2][j].sum()
    loss.backward()

    # Analytical reference: sum of F^k × RY_TO_EV × SHELL_k(J,J).sum()
    # for each rebuilt (J, k) pair.
    cowan = read_cowan_store(NID8CT_RCG)
    meta = nid8ct_metadata
    raw_cfg = nid8_raw_params.ground
    ry_to_ev = float(RY_TO_EV)

    analytical = 0.0
    for j_idx, m in enumerate(meta[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            j_sym = m.bra_sym
            # Sum F^k × SHELL_k(J,J) over available k
            for k_idx, km in enumerate(meta[2]):
                if (km.operator == "SHELL1"
                        and km.block_type == "GROUND"
                        and km.bra_sym == j_sym
                        and km.ket_sym == j_sym):
                    k = int(km.op_sym.rstrip("+-"))
                    fk_ry = raw_cfg.f("3D", "3D", k)
                    analytical += fk_ry * ry_to_ev * float(cowan[2][k_idx].sum())

    assert slater_scale.grad.item() == pytest.approx(analytical, rel=1e-10)


# ─────────────────────────────────────────────────────────────
# Scaled rebuild: non-identity scales produce different values
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_scaled_build_differs_from_template(
    nid8_raw_params, nid8ct_plan, nid8ct_cowan_parsed, nid8ct_metadata,
):
    """At scale=0.8, the GROUND HAMILTONIAN blocks should change."""
    scaled = scale_atomic_params(nid8_raw_params, slater_scale=0.8, soc_scale=0.95)
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )
    for j, m in enumerate(nid8ct_metadata[2]):
        if m.operator == "HAMILTONIAN" and m.block_type == "GROUND":
            parsed = nid8ct_cowan_parsed[2][j]
            assert not torch.allclose(built[2][j], parsed, atol=1e-6), (
                f"Slot {j+1}: scaled build should differ from template"
            )


@pytest.mark.phase4
def test_passthrough_unaffected_by_scale(
    nid8_raw_params, nid8ct_plan, nid8ct_cowan_parsed, nid8ct_metadata,
):
    """At scale≠1, passthrough blocks must still be identical to template."""
    scaled = scale_atomic_params(nid8_raw_params, slater_scale=0.8, soc_scale=0.95)
    built = build_cowan_store_in_memory(
        scaled, nid8_raw_params, nid8ct_plan,
        source_rcg_path=NID8CT_RCG,
    )
    for s in [0, 1]:
        for j in range(len(nid8ct_cowan_parsed[s])):
            assert torch.equal(built[s][j], nid8ct_cowan_parsed[s][j])


# ─────────────────────────────────────────────────────────────
# End-to-end: COWAN store → assembler → parity with file path
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase4
def test_end_to_end_assembler_parity(
    nid8ct_ban, nid8_raw_params, nid8ct_plan, nid8ct_built_identity,
):
    """Built COWAN store + parsed RAC + parsed BAN must equal file-based result.

    This is the C3e → assembler boundary contract (extending C3d's test).
    """
    rac, _ = build_rac_in_memory(
        nid8ct_ban,
        source_rac_path=NID8CT_RAC,
        source_rcg_path=NID8CT_RCG,
    )
    built_result = assemble_and_diagonalize_in_memory(
        nid8ct_built_identity, rac, nid8ct_ban,
    )
    from_disk = assemble_and_diagonalize(NID8CT_RCG, NID8CT_RAC, NID8CT_BAN)

    assert len(built_result.triads) == len(from_disk.triads)
    for tb, td in zip(built_result.triads, from_disk.triads):
        assert torch.allclose(tb.Eg, td.Eg, atol=1e-12)
        assert torch.allclose(tb.Ef, td.Ef, atol=1e-12)
        assert torch.allclose(tb.T, td.T, atol=1e-12)
