"""Unit tests for parity metrics on synthetic spectra."""
from __future__ import annotations

import numpy as np
import pytest

from bench.parity import (
    resample_to_common_grid,
    normalize_unit_max,
    cosine_similarity,
    max_abs_diff,
    peak_position_error,
    l3_l2_ratio,
    compare,
)


def _gaussian(x, mu, sigma, height=1.0):
    return height * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def _ledge_toy(x, l3_center=855.0, l2_center=870.0, width=0.7, height=(1.0, 0.5)):
    """Synthetic L-edge-like spectrum: a big L3 peak + smaller L2 peak."""
    return _gaussian(x, l3_center, width, height[0]) + _gaussian(x, l2_center, width, height[1])


# ─────────────────────────────────────────────────────────────
# Cosine + max diff
# ─────────────────────────────────────────────────────────────


def test_cosine_identical_spectra_is_one():
    x = np.linspace(850.0, 880.0, 3001)
    y = _ledge_toy(x)
    assert abs(cosine_similarity(y, y) - 1.0) < 1e-12


def test_cosine_detects_shape_mismatch():
    x = np.linspace(850.0, 880.0, 3001)
    y1 = _gaussian(x, 855.0, 0.7)
    y2 = _gaussian(x, 855.0, 2.0)
    c = cosine_similarity(y1, y2)
    # Different widths, same position: cosine ~ 0.8-0.95; definitely < 0.9999.
    assert 0.5 < c < 0.99


def test_max_abs_diff_identical_is_zero():
    x = np.linspace(850.0, 880.0, 3001)
    y = _ledge_toy(x)
    y_norm = normalize_unit_max(y)
    assert max_abs_diff(y_norm, y_norm) == 0.0


def test_max_abs_diff_catches_broadening():
    x = np.linspace(850.0, 880.0, 3001)
    y1 = normalize_unit_max(_gaussian(x, 855.0, 0.5))
    y2 = normalize_unit_max(_gaussian(x, 855.0, 2.0))
    d = max_abs_diff(y1, y2)
    # Peak heights normalize to 1.0; broadened wings must differ by O(1) at mid-radius.
    assert d > 0.3


# ─────────────────────────────────────────────────────────────
# Peak position error
# ─────────────────────────────────────────────────────────────


def test_peak_position_error_zero_on_identical():
    x = np.linspace(850.0, 880.0, 3001)
    y = _ledge_toy(x)
    mean_err, max_err, n = peak_position_error(x, y, y, top_k=2)
    assert n >= 1
    assert max_err < 1e-6
    assert mean_err < 1e-6


def test_peak_position_error_detects_shift():
    x = np.linspace(850.0, 880.0, 6001)   # finer grid → sub-0.01eV accuracy
    y1 = _ledge_toy(x, l3_center=855.0, l2_center=870.0)
    y2 = _ledge_toy(x, l3_center=855.2, l2_center=870.3)
    mean_err, max_err, n = peak_position_error(x, y1, y2, top_k=2)
    # Peaks shifted 0.2 and 0.3; expected mean ~0.25, max ~0.3.
    assert 0.15 < mean_err < 0.35
    assert 0.2 <= max_err <= 0.35
    assert n == 2


# ─────────────────────────────────────────────────────────────
# L3/L2 ratio
# ─────────────────────────────────────────────────────────────


def test_l3_l2_ratio_on_ledge_toy():
    # Heights 1.0 (L3) vs 0.5 (L2); integrated ratio should be ~2.0.
    x = np.linspace(850.0, 880.0, 3001)
    y = _ledge_toy(x, height=(1.0, 0.5))
    r = l3_l2_ratio(x, y)
    assert 1.8 < r < 2.2


def test_l3_l2_ratio_symmetric_is_one():
    # Both halves identical → ratio = 1.0.
    x = np.linspace(850.0, 880.0, 3001)
    y = _gaussian(x, 865.0, 3.0)
    r = l3_l2_ratio(x, y)
    assert 0.9 < r < 1.1


# ─────────────────────────────────────────────────────────────
# Common-grid resampling
# ─────────────────────────────────────────────────────────────


def test_resample_takes_intersection_not_union():
    x_a = np.linspace(850.0, 880.0, 101)
    y_a = _ledge_toy(x_a)
    x_b = np.linspace(855.0, 875.0, 51)
    y_b = _ledge_toy(x_b)
    x_common, ya_c, yb_c = resample_to_common_grid(x_a, y_a, x_b, y_b)
    assert x_common.min() >= 855.0 - 1e-9
    assert x_common.max() <= 875.0 + 1e-9
    assert ya_c.shape == yb_c.shape
    assert ya_c.shape == x_common.shape


def test_resample_raises_on_disjoint_ranges():
    x_a = np.linspace(0.0, 10.0, 11)
    x_b = np.linspace(20.0, 30.0, 11)
    y_a = np.ones_like(x_a)
    y_b = np.ones_like(x_b)
    with pytest.raises(ValueError, match="do not overlap"):
        resample_to_common_grid(x_a, y_a, x_b, y_b)


# ─────────────────────────────────────────────────────────────
# End-to-end compare
# ─────────────────────────────────────────────────────────────


def test_compare_end_to_end_on_identical_ledge():
    x = np.linspace(850.0, 880.0, 3001)
    y = _ledge_toy(x, height=(1.0, 0.5))
    r = compare(x, y, x, y, calctype="xas")
    assert r.cosine > 0.9999
    assert r.max_abs_diff < 1e-12
    assert r.peak_err_max_ev < 1e-6
    assert abs(r.l3_l2_ratio - r.l3_l2_ratio_ref) < 1e-12
    assert r.n_peaks_matched >= 1
    assert r.common_grid_n > 100


def test_compare_detects_broadening_only_with_alignment():
    """With peak_align=True, only a lineshape difference remains."""
    x_ref = np.linspace(850.0, 880.0, 6001)
    y_ref = _ledge_toy(x_ref, l3_center=855.0, l2_center=870.0, width=0.7)
    y_broadened = _ledge_toy(x_ref, l3_center=855.0, l2_center=870.0, width=1.4)
    r = compare(x_ref, y_broadened, x_ref, y_ref, calctype="xas")
    # Same centers → small shift; 2× broadening → cosine drops to ~0.89,
    # max_abs_diff ~0.47 (large on unit-max scale). Thresholds below
    # are the measured values for this 2× broadening case.
    assert abs(r.peak_shift_ev) < 0.05
    assert 0.85 < r.cosine < 0.95
    assert r.max_abs_diff > 0.1


def test_compare_captures_bulk_shift_as_peak_shift_ev():
    """A pure rigid shift is absorbed by peak_shift_ev and metrics look identical.

    Sign convention: peak_shift_ev = argmax(b) − argmax(a), so when a's
    peak is to the right of b's peak (y_shifted.center=855.2 > y_ref.center=855.0),
    peak_shift_ev = -0.2 (a is shifted LEFT by 0.2 to align with b).
    """
    x_ref = np.linspace(850.0, 880.0, 6001)
    y_ref = _ledge_toy(x_ref, l3_center=855.0, l2_center=870.0)
    y_shifted = _ledge_toy(x_ref, l3_center=855.2, l2_center=870.2)
    r = compare(x_ref, y_shifted, x_ref, y_ref, calctype="xas")
    assert abs(r.peak_shift_ev - (-0.2)) < 0.02
    # After alignment the spectra match.
    assert r.cosine > 0.9999
    assert r.max_abs_diff < 1e-6


def test_compare_peak_align_false_still_sees_shift():
    """With alignment disabled the old behaviour — peak error reflects the shift."""
    x_ref = np.linspace(850.0, 880.0, 6001)
    y_ref = _ledge_toy(x_ref, l3_center=855.0, l2_center=870.0, width=0.7)
    y_shifted = _ledge_toy(x_ref, l3_center=855.2, l2_center=870.3, width=0.7)
    r = compare(x_ref, y_shifted, x_ref, y_ref, calctype="xas", peak_align=False)
    assert not r.peak_aligned
    assert r.peak_err_max_ev > 0.05
    assert r.cosine < 0.9999
