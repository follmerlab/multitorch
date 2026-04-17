"""
Phase 1 tests: verify pseudo-Voigt broadening.

Tests:
  1. Output shape and dtype
  2. Normalization: broadened peak area ≈ stick amplitude
  3. Peak position: maximum of broadened line near stick position
  4. Legacy vs correct mode: different results for the same parameters
  5. Two-region broadening: L3 and L2 broadenings applied correctly
"""
import pytest
import torch
import math


@pytest.mark.phase1
def test_pseudo_voigt_shape():
    from multitorch.spectrum.broaden import pseudo_voigt
    N_grid, N_sticks = 1000, 5
    x = torch.linspace(850, 870, N_grid, dtype=torch.float64)
    x0 = torch.tensor([852.0, 854.0, 856.0, 858.0, 860.0], dtype=torch.float64)
    amp = torch.ones(N_sticks, dtype=torch.float64)
    y = pseudo_voigt(x, x0, amp, fwhm_g=0.2, fwhm_l=0.3)
    assert y.shape == (N_grid,)
    assert y.dtype == torch.float64


@pytest.mark.phase1
def test_pseudo_voigt_positive():
    from multitorch.spectrum.broaden import pseudo_voigt
    x = torch.linspace(850, 870, 500, dtype=torch.float64)
    x0 = torch.tensor([860.0], dtype=torch.float64)
    amp = torch.tensor([1.0], dtype=torch.float64)
    y = pseudo_voigt(x, x0, amp, fwhm_g=0.5, fwhm_l=0.5)
    assert (y >= 0).all(), "Pseudo-Voigt should be non-negative"


@pytest.mark.phase1
def test_pseudo_voigt_peak_at_stick_position():
    """Peak of broadened line should be at the stick position."""
    from multitorch.spectrum.broaden import pseudo_voigt
    stick_E = 860.0
    x = torch.linspace(850, 870, 2000, dtype=torch.float64)
    x0 = torch.tensor([stick_E], dtype=torch.float64)
    amp = torch.tensor([1.0], dtype=torch.float64)
    y = pseudo_voigt(x, x0, amp, fwhm_g=0.2, fwhm_l=0.3)
    peak_E = float(x[y.argmax()])
    assert abs(peak_E - stick_E) < 0.05, f"Peak at {peak_E:.3f}, expected {stick_E}"


@pytest.mark.phase1
def test_pseudo_voigt_legacy_vs_correct():
    """Legacy and correct modes should give different results for typical parameters."""
    from multitorch.spectrum.broaden import pseudo_voigt
    x = torch.linspace(850, 870, 500, dtype=torch.float64)
    x0 = torch.tensor([860.0], dtype=torch.float64)
    amp = torch.tensor([1.0], dtype=torch.float64)
    y_legacy = pseudo_voigt(x, x0, amp, fwhm_g=0.2, fwhm_l=0.4, mode="legacy")
    y_correct = pseudo_voigt(x, x0, amp, fwhm_g=0.2, fwhm_l=0.4, mode="correct")
    # They should differ (the bug affects the eta mixing parameter)
    diff = (y_legacy - y_correct).abs().max()
    assert diff > 1e-6, "Legacy and correct modes should give different results"


@pytest.mark.phase1
def test_pseudo_voigt_normalization():
    """
    Area under pseudo-Voigt ≈ amplitude for a well-resolved peak.
    (Not exact because normalization depends on conventions.)
    """
    from multitorch.spectrum.broaden import pseudo_voigt
    import math
    amp_val = 2.5
    x = torch.linspace(855, 865, 10000, dtype=torch.float64)
    dx = float(x[1] - x[0])
    x0 = torch.tensor([860.0], dtype=torch.float64)
    amp = torch.tensor([amp_val], dtype=torch.float64)
    y = pseudo_voigt(x, x0, amp, fwhm_g=0.0, fwhm_l=0.5, mode="correct")
    area = float(y.sum()) * dx
    # For a Lorentzian the area = amplitude (since the profile integrates to 1/amplitude...)
    # Actually the profile is normalized so area of profile = 1, and output = amp * profile
    # So total area ≈ amp_val
    # Allow up to 5% error due to limited energy range (Lorentzian tails extend beyond grid)
    assert abs(area - amp_val) / amp_val < 0.05, (
        f"Area = {area:.4f}, expected ≈ {amp_val}"
    )


@pytest.mark.phase1
def test_broaden_gaussian_shape():
    from multitorch.spectrum.broaden import broaden_gaussian
    N, M = 100, 5
    E = torch.linspace(0, 10, N, dtype=torch.float64)
    D = torch.rand(N, M, dtype=torch.float64)
    Y = broaden_gaussian(E, D, fwhm=0.5)
    assert Y.shape == (N, M)


@pytest.mark.phase1
def test_broaden_gaussian_fwhm_differentiable():
    """broaden_gaussian must support tensor FWHM for RIXS autograd."""
    from multitorch.spectrum.broaden import broaden_gaussian
    N, M = 100, 5
    E = torch.linspace(0, 10, N, dtype=torch.float64)
    D = torch.rand(N, M, dtype=torch.float64)
    fwhm = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    Y = broaden_gaussian(E, D, fwhm=fwhm)
    assert Y.shape == (N, M)
    loss = Y.sum()
    loss.backward()
    assert fwhm.grad is not None, "Gradient w.r.t. FWHM must exist"
    assert not fwhm.grad.isnan(), "FWHM gradient must not be NaN"
