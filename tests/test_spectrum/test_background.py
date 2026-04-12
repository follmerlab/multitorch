"""
Tests for multitorch.spectrum.background — XAS background and saturation.
"""
from __future__ import annotations

import math

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.spectrum.background import add_background, _arctangent_step, saturate


# ─────────────────────────────────────────────────────────────
# _arctangent_step
# ─────────────────────────────────────────────────────────────

def test_arctangent_step_at_edge_is_half_height():
    """Step function at E = E0 should be height/2."""
    E = torch.tensor([850.0], dtype=DTYPE)
    step = _arctangent_step(E, E0=850.0, height=2.0, width=0.5)
    assert torch.allclose(step, torch.tensor([1.0], dtype=DTYPE), atol=1e-10)


def test_arctangent_step_limits():
    """Far below edge → ~0, far above → ~height."""
    E = torch.linspace(800.0, 900.0, 1000, dtype=DTYPE)
    step = _arctangent_step(E, E0=850.0, height=1.0, width=0.5)
    assert step[0] < 0.01      # well below edge
    assert step[-1] > 0.99     # well above edge


def test_arctangent_step_is_monotonic():
    """Step function must be monotonically increasing."""
    E = torch.linspace(845.0, 855.0, 500, dtype=DTYPE)
    step = _arctangent_step(E, E0=850.0, height=1.0, width=0.5)
    diffs = step[1:] - step[:-1]
    assert (diffs >= 0).all()


def test_arctangent_step_width_controls_sharpness():
    """Narrower width → sharper step (larger derivative at E0)."""
    E = torch.linspace(849.0, 851.0, 100, dtype=DTYPE)
    step_narrow = _arctangent_step(E, E0=850.0, height=1.0, width=0.1)
    step_wide = _arctangent_step(E, E0=850.0, height=1.0, width=2.0)
    # Derivative at center: height / (π * width)
    # Narrow should have steeper rise
    deriv_narrow = (step_narrow[51] - step_narrow[49]).item()
    deriv_wide = (step_wide[51] - step_wide[49]).item()
    assert deriv_narrow > deriv_wide


# ─────────────────────────────────────────────────────────────
# add_background
# ─────────────────────────────────────────────────────────────

def test_add_background_step_only():
    """With only step, output = I + step."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.ones_like(E)
    result = add_background(E, I, edge_energy=850.0, step_height=2.0, width=0.5)
    assert result.shape == E.shape
    # At E=850, step = height/2 = 1.0, so result = I + step ≈ 2.0
    mid = (E - 850.0).abs().argmin()
    assert abs(result[mid].item() - 2.0) < 0.1


def test_add_background_linear_slope():
    """Linear component adds slope * (E - E0)."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.zeros_like(E)
    result = add_background(
        E, I, edge_energy=850.0, step_height=0.0, width=0.5,
        linear_slope=0.1,
    )
    # At E=855, linear contrib = 0.1 * 5 = 0.5
    assert abs(result[-1].item() - 0.5) < 0.1


def test_add_background_gaussian():
    """Gaussian pre-edge feature appears at specified center."""
    E = torch.linspace(845.0, 855.0, 1000, dtype=DTYPE)
    I = torch.zeros_like(E)
    result = add_background(
        E, I, edge_energy=850.0, step_height=0.0, width=0.5,
        gaussian_amp=1.0, gaussian_center=848.0, gaussian_fwhm=0.5,
    )
    # Peak should be near E=848
    peak_idx = int(torch.argmax(result))
    peak_E = E[peak_idx].item()
    assert abs(peak_E - 848.0) < 0.1


def test_add_background_preserves_dtype_and_shape():
    """Output has same dtype and shape as input."""
    E = torch.linspace(845.0, 855.0, 200, dtype=DTYPE)
    I = torch.randn(200, dtype=DTYPE)
    result = add_background(E, I, edge_energy=850.0)
    assert result.shape == I.shape
    assert result.dtype == I.dtype


# ─────────────────────────────────────────────────────────────
# saturate
# ─────────────────────────────────────────────────────────────

def test_saturate_zero_thickness_is_identity():
    """At thickness → 0, saturation correction → 1 (no change)."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.ones_like(E) * 0.5
    result = saturate(E, I, thickness=1e-8, independent=True)
    # (1 - exp(-ε)) / ε ≈ 1 for small ε
    assert torch.allclose(result, I, atol=1e-4)


def test_saturate_reduces_peaks():
    """Saturation should reduce peak intensity relative to baseline."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.zeros_like(E)
    I[50] = 10.0   # sharp peak
    I[40] = 0.1    # low-intensity point
    result = saturate(E, I, thickness=1.0, independent=True)
    # Peak is reduced more than baseline
    ratio_peak = result[50] / I[50]
    ratio_base = result[40] / I[40]
    assert ratio_peak < ratio_base


def test_saturate_all_finite():
    """No NaN or inf in output, even with zeros in input."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.zeros_like(E)
    I[30:70] = 1.0
    result = saturate(E, I, thickness=2.0, independent=True)
    assert torch.isfinite(result).all()


def test_saturate_angle_dependent():
    """Non-independent mode uses angle; 90° should differ from 45°."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.ones_like(E)
    r45 = saturate(E, I, thickness=1.0, angle=45.0, independent=False)
    r90 = saturate(E, I, thickness=1.0, angle=90.0, independent=False)
    # sin(90°)=1 vs sin(45°)≈0.707 → different effective thickness
    assert not torch.allclose(r45, r90)


def test_saturate_independent_flag():
    """Independent mode ignores angle."""
    E = torch.linspace(845.0, 855.0, 100, dtype=DTYPE)
    I = torch.ones_like(E)
    r1 = saturate(E, I, thickness=1.0, angle=45.0, independent=True)
    r2 = saturate(E, I, thickness=1.0, angle=90.0, independent=True)
    assert torch.allclose(r1, r2)
