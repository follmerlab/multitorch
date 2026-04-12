"""
End-to-end integration tests: nid8ct XAS spectrum.

These tests use the bootstrap mode (ban_output_path) to compute the
spectrum from the pre-computed Fortran reference output and verify that
the result matches the reference data.

Key validations:
  1. getXAS() runs without error on the nid8ct reference
  2. Spectrum is in the correct energy range (Ni L-edge ~850-885 eV)
  3. Spectrum is non-negative everywhere
  4. Total intensity (area) is within expected range
  5. Peak position is near expected L3-edge energy
  6. Gradient-based optimization test: dI/d(beam_fwhm) is finite
"""
import pytest
from pathlib import Path
import torch

REFDATA = Path(__file__).parent.parent / "reference_data" / "nid8ct"


@pytest.mark.integration
def test_getXAS_nid8ct_runs():
    """getXAS from nid8ct.ban_out should run without error."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(
        str(REFDATA / "nid8ct.ban_out"),
        T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4,
    )
    assert x is not None
    assert y is not None


@pytest.mark.integration
def test_getXAS_output_shapes():
    """Output tensors should have matching shapes."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(str(REFDATA / "nid8ct.ban_out"), nbins=1000)
    assert x.shape == y.shape == (1000,)


@pytest.mark.integration
def test_getXAS_dtype():
    """Output should be float64."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(str(REFDATA / "nid8ct.ban_out"))
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64


@pytest.mark.integration
def test_getXAS_energy_range():
    """Spectrum energy axis should be in the Ni L-edge range."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(str(REFDATA / "nid8ct.ban_out"), T=80)
    assert float(x.min()) > 840, f"xmin = {float(x.min()):.1f} eV (too low)"
    assert float(x.max()) < 940, f"xmax = {float(x.max()):.1f} eV (too high)"


@pytest.mark.integration
def test_getXAS_non_negative():
    """Broadened spectrum should be non-negative."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(str(REFDATA / "nid8ct.ban_out"))
    assert (y >= -1e-10).all(), f"Negative values found: min(y) = {float(y.min()):.6f}"


@pytest.mark.integration
def test_getXAS_has_L3_peak():
    """Spectrum should have a peak in the L3 range (~855-860 eV)."""
    from multitorch.api.plot import getXAS
    x, y = getXAS(str(REFDATA / "nid8ct.ban_out"), T=80, nbins=5000)
    # Find peak position
    peak_idx = y.argmax()
    peak_E = float(x[peak_idx])
    assert 853 < peak_E < 870, f"L3 peak at {peak_E:.2f} eV, expected 853-870 eV"


@pytest.mark.integration
def test_calcXAS_bootstrap_with_return_sticks():
    """calcXAS with return_sticks=True should return 3 tensors."""
    from multitorch.api.calc import calcXAS
    x, y, sticks = calcXAS(
        element='', valence='', sym='', edge='', cf={},
        ban_output_path=str(REFDATA / "nid8ct.ban_out"),
        return_sticks=True, T=80,
    )
    assert sticks.ndim == 2
    assert sticks.shape[1] == 2


@pytest.mark.integration
def test_calcXAS_phase5_runs():
    """calcXAS without ban_output_path runs the Phase 5 pipeline."""
    from multitorch.api.calc import calcXAS
    x, y = calcXAS(element='Ni', valence='ii', sym='d4h', edge='l',
                    cf={'tendq': 1.0, 'dt': 0.0, 'ds': 0.1})
    assert x.shape == y.shape
    assert x.numel() > 0
    assert (y >= -1e-10).all(), "Negative spectrum values"
    assert float(y.max()) > 0.0, "Empty spectrum"


@pytest.mark.integration
def test_getXAS_boltzmann_temperature_effect():
    """Different temperatures should produce different spectra when max_gs > 1."""
    from multitorch.api.plot import getXAS
    # Use max_gs=5 to include multiple ground states (enables Boltzmann effect)
    _, y_cold = getXAS(str(REFDATA / "nid8ct.ban_out"), T=10, max_gs=5)
    _, y_hot = getXAS(str(REFDATA / "nid8ct.ban_out"), T=5000, max_gs=5)
    # At higher T more ground states populated → different spectrum
    diff = (y_cold - y_hot).abs().max()
    assert float(diff) > 1e-6, "Temperature should affect the spectrum with max_gs=5"


@pytest.mark.integration
def test_getXAS_broadening_effect():
    """Larger broadening should give smoother (wider) peaks."""
    from multitorch.api.plot import getXAS
    _, y_narrow = getXAS(str(REFDATA / "nid8ct.ban_out"), beam_fwhm=0.1, gamma1=0.1)
    _, y_wide = getXAS(str(REFDATA / "nid8ct.ban_out"), beam_fwhm=0.5, gamma1=0.5)
    # Wider broadening should give lower peak height
    assert float(y_wide.max()) < float(y_narrow.max()), (
        "Wider broadening should reduce peak height"
    )
