"""
Phase 0 tests: verify RME file parsers (read_rme_rcg and read_rme_rac).

Tests verify that:
  1. Parsers run without exception on all three reference examples
  2. IRREP metadata is correctly extracted
  3. Key RME matrix values match reference files exactly
  4. Matrix tensors are float64

Reference values taken from:
  tests/reference_data/nid8ct/nid8ct.rme_rcg
  tests/reference_data/nid8ct/nid8ct.rme_rac
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data"


# ── .rme_rcg parser ──────────────────────────────────────────

@pytest.mark.phase0
def test_read_rme_rcg_nid8ct(nid8ct_rme_rcg):
    assert nid8ct_rme_rcg is not None
    assert len(nid8ct_rme_rcg.configs) >= 1


@pytest.mark.phase0
def test_rme_rcg_has_irreps(nid8ct_rme_rcg):
    cfg = nid8ct_rme_rcg.configs[0]
    assert len(cfg.irreps) > 0


@pytest.mark.phase0
def test_rme_rcg_irrep_names(nid8ct_rme_rcg):
    """First config should have GROUND irreps 0+, 1+, 2+, 3+, 4+."""
    cfg = nid8ct_rme_rcg.configs[0]
    ground_irreps = [ir.name for ir in cfg.irreps if ir.kind == "GROUND"]
    assert "0+" in ground_irreps
    assert "1+" in ground_irreps


@pytest.mark.phase0
def test_rme_rcg_has_multipole_blocks(nid8ct_rme_rcg):
    """Should have at least one MULTIPOLE transition block."""
    cfg = nid8ct_rme_rcg.configs[0]
    multipole_blocks = [b for b in cfg.blocks if b.operator == "MULTIPOLE"]
    assert len(multipole_blocks) > 0, "No MULTIPOLE blocks found"


@pytest.mark.phase0
def test_rme_rcg_first_block_matrix(nid8ct_rme_rcg):
    """
    From nid8ct.rme_rcg:
      RME TRANSITION  0+  1-  1-  MULTIPOLE  2  3  3
        1  2  ,  1  0.316228 ,  2  -0.547723 ;
        2  1  ,  3  0.632456 ;
    Check matrix shape and a specific value.
    """
    cfg = nid8ct_rme_rcg.configs[0]
    block = cfg.get_block("0+", "1-", "1-", "MULTIPOLE")
    assert block is not None, "Block (0+, 1-, 1-, MULTIPOLE) not found"
    assert block.matrix.shape == (2, 3)
    assert block.matrix.dtype == torch.float64
    # Row 0 (index 0), col 0 (1-indexed col 1): 0.316228
    torch.testing.assert_close(
        block.matrix[0, 0],
        torch.tensor(0.316228, dtype=torch.float64),
        atol=1e-5, rtol=0,
    )
    # Row 0, col 1 (1-indexed col 2): -0.547723
    torch.testing.assert_close(
        block.matrix[0, 1],
        torch.tensor(-0.547723, dtype=torch.float64),
        atol=1e-5, rtol=0,
    )


@pytest.mark.phase0
def test_rme_rcg_nid8(nid8_rme_rcg):
    """nid8 single-config CF calculation has SHELL2/HAMILTONIAN blocks (not MULTIPOLE)."""
    assert len(nid8_rme_rcg.configs) >= 1
    # nid8 is CF-only (no transition .m14); it has SHELL2/HAMILTONIAN blocks
    assert len(nid8_rme_rcg.configs[0].blocks) > 0


@pytest.mark.phase0
def test_rme_rcg_als1ni2(als1ni2_rme_rcg):
    assert len(als1ni2_rme_rcg.configs) >= 1


# ── .rme_rac parser ──────────────────────────────────────────

@pytest.mark.phase0
def test_read_rme_rac_nid8ct(nid8ct_rme_rac):
    assert nid8ct_rme_rac is not None
    assert len(nid8ct_rme_rac.blocks) > 0


@pytest.mark.phase0
def test_rme_rac_irreps(nid8ct_rme_rac):
    """nid8ct.rme_rac has both regular and '^'-prefixed irreps."""
    names = [ir.name for ir in nid8ct_rme_rac.irreps]
    assert "0+" in names
    assert "^0+" in names


@pytest.mark.phase0
def test_rme_rac_transi_blocks(nid8ct_rme_rac):
    """Should have TRANSI blocks for the allowed (sym1, sym2, sym3) triads."""
    transi = [b for b in nid8ct_rme_rac.blocks if b.kind == "TRANSI"]
    assert len(transi) > 0


@pytest.mark.phase0
def test_rme_rac_first_transi_shape(nid8ct_rme_rac):
    """
    From nid8ct.rme_rac:
      REDUCEDMATRIX TRANSI  0+  1-  1-  PERP  9  15
    Block should have shape (9, 15).
    """
    block = nid8ct_rme_rac.get_block("0+", "1-", "1-", "TRANSI")
    assert block is not None, "TRANSI block (0+, 1-, 1-) not found"
    assert block.matrix.shape == (9, 15), (
        f"Expected (9, 15), got {block.matrix.shape}"
    )
    assert block.matrix.dtype == torch.float64


@pytest.mark.phase0
def test_rme_rac_nid8_loads_without_error(nid8_rme_rac):
    """nid8 uses butler mode (.ora format) — rme_rac returns empty but no crash."""
    assert nid8_rme_rac is not None
    # nid8 uses butler mode; rme_rac is .ora format, returns empty RACFile
    # REDUCEDMATRIX blocks will be populated once .ora parser is added (Phase 3)


@pytest.mark.phase0
def test_rme_rac_als1ni2_loads_without_error(als1ni2_rme_rac):
    """als1ni2 uses butler mode (.ora format) — rme_rac returns empty but no crash."""
    assert als1ni2_rme_rac is not None
