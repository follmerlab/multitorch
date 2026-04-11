"""
Phase 1 tests: verify the Boltzmann-weighted stick spectrum generation.

Given nid8ct.ban_out reference data, verify that get_sticks() produces
energies and intensities consistent with the Fortran output.

Key reference values from nid8ct.ban_out (triad 0+, 1-, 1-):
  Ground state: Eg = -2.19045 Ry
  First final state: Ef = 852.30042 eV
  First intensity: M[0,0] = 0.043971
  Total row intensity: 0.27102
"""
import pytest
import torch
from pathlib import Path

REFDATA = Path(__file__).parent.parent / "reference_data"


@pytest.mark.phase1
def test_get_sticks_runs_without_error(nid8ct_ban):
    from multitorch.spectrum.sticks import get_sticks
    Etrans, Mtrans, Eg_min = get_sticks(nid8ct_ban, T=0)
    assert Etrans.numel() > 0
    assert Mtrans.numel() > 0


@pytest.mark.phase1
def test_get_sticks_dtype(nid8ct_ban):
    from multitorch.spectrum.sticks import get_sticks
    Etrans, Mtrans, _ = get_sticks(nid8ct_ban, T=0)
    assert Etrans.dtype == torch.float64
    assert Mtrans.dtype == torch.float64


@pytest.mark.phase1
def test_get_sticks_no_negative_intensities_at_T0(nid8ct_ban):
    """At T=0 with pre-squared intensities, all M should be >= 0."""
    from multitorch.spectrum.sticks import get_sticks
    _, Mtrans, _ = get_sticks(nid8ct_ban, T=0)
    assert (Mtrans >= -1e-10).all(), "Negative intensities found"


@pytest.mark.phase1
def test_get_sticks_energies_in_xas_range(nid8ct_ban):
    """
    Transition energies should be in the Ni L-edge range.

    pyctm convention: Etrans = Ef_eV - Eg_Ry (mixed units). With Ef ≈ 852 eV
    and Eg ≈ -2.19 Ry, Etrans ≈ 854 eV. Range extended to account for all
    charge-transfer configurations (CT states extend above 876 eV) and the
    Ry correction.
    """
    from multitorch.spectrum.sticks import get_sticks
    Etrans, _, _ = get_sticks(nid8ct_ban, T=0)
    assert float(Etrans.min()) > 845, "Minimum energy too low for Ni L-edge"
    assert float(Etrans.max()) < 930, "Maximum energy unreasonably high"


@pytest.mark.phase1
def test_get_sticks_first_transition_energy(nid8ct_ban):
    """
    Transition energies using pyctm convention: Etrans = Ef_eV - Eg_Ry.
    For nid8ct CT calculation, max_gs=1 selects the globally lowest ground
    state (in the 1+ symmetry sector at ~-3.45 Ry). First transition:
    852.18597 - (-3.45060) ≈ 855.64 eV.
    """
    from multitorch.spectrum.sticks import get_sticks
    Etrans, _, _ = get_sticks(nid8ct_ban, T=0, max_gs=1)
    min_E = float(Etrans.min())
    # Should be near 855-856 eV region (from the J=1+ ground state sector)
    assert abs(min_E - 855.5) < 1.5, f"First stick at {min_E:.2f} eV, expected ~855.5 eV"


@pytest.mark.phase1
def test_get_sticks_boltzmann_effect(nid8ct_ban):
    """Boltzmann weighting at T=300K should not change total intensity dramatically."""
    from multitorch.spectrum.sticks import get_sticks
    _, M0, _ = get_sticks(nid8ct_ban, T=0)
    _, M300, _ = get_sticks(nid8ct_ban, T=300)
    # Total intensity ratio should be in [0.5, 2.0]
    ratio = M300.sum() / (M0.sum() + 1e-10)
    assert 0.5 < float(ratio) < 2.0, f"T=300K / T=0 ratio = {ratio:.3f}, unexpected"
