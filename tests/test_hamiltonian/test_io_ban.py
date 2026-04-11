"""
Phase 0 tests: verify the .ban_out / .oba parser (read_ban_output).

Tests verify that:
  1. Parser runs without exception on nid8ct reference data
  2. Triad structure is correctly identified
  3. Key numeric values match the reference file exactly
  4. Tensors are float64

Reference values from:
  tests/reference_data/nid8ct/nid8ct.ban_out

Key reference values for first triad (0+, 1-, 1-):
  Ground state energy: Eg = -2.19045 Ry
  First final state energy: Ef = 852.30042 eV
  First matrix element M[0,0] = 0.043971
  Total intensity (sum of row): 0.27102
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data"


@pytest.mark.phase0
def test_read_ban_output_no_error(nid8ct_ban):
    assert nid8ct_ban is not None


@pytest.mark.phase0
def test_ban_has_multipole_actor(nid8ct_ban):
    assert "MULTIPOLE" in nid8ct_ban.triads


@pytest.mark.phase0
def test_ban_first_triad_exists(nid8ct_ban):
    """Triad (0+, 1-, 1-) must be present under MULTIPOLE actor."""
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block is not None, "Triad (0+, 1-, 1-) not found under MULTIPOLE"


@pytest.mark.phase0
def test_ban_ground_state_energy(nid8ct_ban):
    """
    From nid8ct.ban_out:
      Ground state energy Eg0 = -2.19045 Ry (for first config)
    """
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block is not None
    # Eg is in Ry; first entry should be ~-2.19045
    eg0 = block.Eg[0].item()
    assert abs(eg0 - (-2.19045)) < 0.001, (
        f"Ground state energy {eg0:.5f}, expected ~-2.19045 Ry"
    )


@pytest.mark.phase0
def test_ban_first_final_state_energy(nid8ct_ban):
    """
    From nid8ct.ban_out:
      BRA/KET : 852.30042 ...
    First final state should be ~852.30042 eV.
    """
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block is not None
    ef0 = block.Ef[0].item()
    assert abs(ef0 - 852.30042) < 0.01, (
        f"First Ef = {ef0:.5f}, expected ~852.30042 eV"
    )


@pytest.mark.phase0
def test_ban_matrix_element_value(nid8ct_ban):
    """
    From nid8ct.ban_out first triad, row 0, column 0:
      -2.19045: 0.043971
    """
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block is not None
    m00 = block.M[0, 0].item()
    assert abs(m00 - 0.043971) < 5e-4, (
        f"M[0,0] = {m00:.6f}, expected ~0.043971"
    )


@pytest.mark.phase0
def test_ban_total_intensity(nid8ct_ban):
    """
    Total intensity for (0+, 1-, 1-) = 0.27102 (from ban_out).
    Sum of first ground state row should match.
    """
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block is not None
    row_sum = block.M[0].sum().item()
    assert abs(row_sum - 0.27102) < 0.005, (
        f"Row sum = {row_sum:.5f}, expected ~0.27102"
    )


@pytest.mark.phase0
def test_ban_tensor_dtype(nid8ct_ban):
    """All tensors should be float64."""
    block = nid8ct_ban.get("MULTIPOLE", ("0+", "1-", "1-"))
    assert block.Eg.dtype == torch.float64
    assert block.Ef.dtype == torch.float64
    assert block.M.dtype == torch.float64


@pytest.mark.phase0
def test_ban_multiple_triads(nid8ct_ban):
    """nid8ct should have multiple triads (several (gs, op, fs) combinations)."""
    all_triads = list(nid8ct_ban.all_triads())
    assert len(all_triads) > 3, (
        f"Expected many triads, got {len(all_triads)}"
    )
