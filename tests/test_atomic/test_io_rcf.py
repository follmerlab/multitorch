"""
Phase 0 tests: verify that the rcn31 and rcn2 output parsers load
reference data correctly.

These tests do NOT test any physics — they only verify that:
  1. The parser runs without exception
  2. Key numeric values match what is in the reference file
  3. Data is returned as torch.float64 tensors

Reference values taken directly from:
  tests/reference_data/nid8/nid8.rcn31_out
  tests/reference_data/nid8/nid8.rcn2_out
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data"


@pytest.mark.phase0
def test_read_rcn31_loads_without_error(nid8_hfs):
    assert nid8_hfs is not None
    assert len(nid8_hfs) >= 1


@pytest.mark.phase0
def test_rcn31_first_config_label(nid8_hfs):
    """First config should be the ground state Ni2+ 2p06 3d08."""
    label = nid8_hfs[0].config_label
    assert "Ni" in label or "ni" in label.lower()


@pytest.mark.phase0
def test_rcn31_has_2p_and_3d_orbitals(nid8_hfs):
    gs = nid8_hfs[0]
    assert gs.orbital("2P") is not None, "2P orbital not found in rcn31 output"
    assert gs.orbital("3D") is not None, "3D orbital not found in rcn31 output"


@pytest.mark.phase0
def test_rcn31_2p_soc_value(nid8_hfs):
    """
    From nid8.rcn31_out:
      2P  6.  0.81583  11.100  (Blume-Watson Ry, eV)
    The 2p SOC should be ~11.100 eV.
    """
    gs = nid8_hfs[0]
    orb = gs.orbital("2P")
    assert orb is not None
    assert abs(orb.zeta_ev - 11.100) < 0.01, (
        f"2P SOC = {orb.zeta_ev:.3f} eV, expected ~11.100 eV"
    )


@pytest.mark.phase0
def test_rcn31_3d_energy_negative(nid8_hfs):
    """3D orbital energy should be negative (bound state)."""
    gs = nid8_hfs[0]
    orb = gs.orbital("3D")
    assert orb is not None
    assert orb.ee_ry < 0.0, f"3D energy {orb.ee_ry} should be negative"


@pytest.mark.phase0
def test_read_rcn2_loads_without_error(nid8_slater):
    assert nid8_slater is not None


@pytest.mark.phase0
def test_rcn2_returns_list(nid8_slater):
    assert isinstance(nid8_slater, list)
    assert len(nid8_slater) >= 1


@pytest.mark.phase0
def test_rcn2_fk_dd_is_tensor(nid8_slater):
    """Slater Fk(dd) should be a float64 tensor."""
    sp = nid8_slater[0]
    assert isinstance(sp.Fk_dd, torch.Tensor)
    assert sp.Fk_dd.dtype == torch.float64


@pytest.mark.phase0
def test_rcn2_f2dd_value(nid8_slater):
    """
    From nid8.rcn2_out for Ni2+ 2p06 3d08:
      F2(dd) ≈ 7.598 eV, F4(dd) ≈ 0.083 eV
    """
    sp = nid8_slater[0]
    f2dd = sp.Fk_dd[1].item()
    assert abs(f2dd - 7.598) < 0.1, f"F2(dd) = {f2dd:.3f}, expected ~7.598 eV"
