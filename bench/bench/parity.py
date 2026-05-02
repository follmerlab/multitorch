"""Spectral parity metrics on a common interpolated grid.

All metrics are computed between two (x, y) spectra after resampling
both onto a common linear grid. The reference impl (ttmult_raw in the
full run) is the right-hand side of every comparison.

Metrics:

1. ``cosine_similarity`` — cosine of angle between intensity vectors.
   Invariant to uniform scaling. Target ≥ 0.9999 for multitorch vs
   Fortran.

2. ``max_abs_diff`` — L∞ norm of the intensity difference after
   normalizing both spectra to unit max. Catches lineshape mismatches
   that cosine doesn't (e.g. a broadened-vs-sharp pair can be cosine-
   similar but look different).

3. ``peak_position_error_ev`` — matches top-K peaks via
   ``scipy.signal.find_peaks``, pairs each a-peak to its nearest
   b-peak, reports mean and max |ΔeV|.

4. ``l3_l2_ratio`` — integrated intensity ratio for L-edge pairs. The
   L3 / L2 windows are derived from the full x-range: L3 is the lower
   half, L2 the upper. Good enough for L2,3-edge benchmarking of 3d
   transition metals; for K-edge or M-edge pass ``l3_window`` /
   ``l2_window`` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from bench.config import (
    COMMON_GRID_SPACING_EV,
    PEAK_TOP_K_LEDGE,
    PEAK_TOP_K_RIXS,
)


@dataclass
class ParityResult:
    cosine: float
    max_abs_diff: float
    peak_err_mean_ev: float
    peak_err_max_ev: float
    l3_l2_ratio: float
    l3_l2_ratio_ref: float
    n_peaks_matched: int
    common_grid_n: int
    common_grid_min: float
    common_grid_max: float
    # Shift applied to spectrum `a` to align its argmax onto spectrum `b`'s
    # argmax. Different CTM codes (multitorch, pyctm, ttmult_raw) use
    # different energy zeros, so an offset is expected and not a bug.
    # Users read this to confirm the offset is a rigid shift, not a
    # lineshape distortion.
    peak_shift_ev: float = 0.0
    peak_aligned: bool = False

    def as_dict(self) -> dict:
        return {
            "cosine": self.cosine,
            "max_abs_diff": self.max_abs_diff,
            "peak_err_mean_ev": self.peak_err_mean_ev,
            "peak_err_max_ev": self.peak_err_max_ev,
            "l3_l2_ratio": self.l3_l2_ratio,
            "l3_l2_ratio_ref": self.l3_l2_ratio_ref,
            "n_peaks_matched": self.n_peaks_matched,
            "common_grid_n": self.common_grid_n,
            "common_grid_min": self.common_grid_min,
            "common_grid_max": self.common_grid_max,
            "peak_shift_ev": self.peak_shift_ev,
            "peak_aligned": self.peak_aligned,
        }


# ─────────────────────────────────────────────────────────────
# Interpolation + normalization
# ─────────────────────────────────────────────────────────────


def resample_to_common_grid(
    x_a: np.ndarray, y_a: np.ndarray,
    x_b: np.ndarray, y_b: np.ndarray,
    spacing_ev: float = COMMON_GRID_SPACING_EV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both spectra onto the intersection of their x-ranges.

    The intersection is the safer default: it avoids extrapolation
    artifacts at the edges of whichever impl emits a narrower range.
    Returns (x_common, y_a_common, y_b_common).
    """
    x_min = max(float(x_a.min()), float(x_b.min()))
    x_max = min(float(x_a.max()), float(x_b.max()))
    if not (x_max > x_min):
        raise ValueError(
            f"Spectra do not overlap: a=[{x_a.min()}, {x_a.max()}], "
            f"b=[{x_b.min()}, {x_b.max()}]"
        )
    n = int(np.floor((x_max - x_min) / spacing_ev)) + 1
    x_common = np.linspace(x_min, x_max, n)
    y_a_common = np.interp(x_common, x_a, y_a)
    y_b_common = np.interp(x_common, x_b, y_b)
    return x_common, y_a_common, y_b_common


def normalize_unit_max(y: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(y)))
    if m == 0.0:
        return y.copy()
    return y / m


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────


def cosine_similarity(y_a: np.ndarray, y_b: np.ndarray) -> float:
    na = float(np.linalg.norm(y_a))
    nb = float(np.linalg.norm(y_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(y_a, y_b) / (na * nb))


def max_abs_diff(y_a: np.ndarray, y_b: np.ndarray) -> float:
    return float(np.max(np.abs(y_a - y_b)))


def _top_k_peaks(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Return x-coordinates of the top-K peaks by height, ascending."""
    from scipy.signal import find_peaks
    # Require at least 3 pts separation to avoid noise; height prominence
    # is relative to the max.
    prominence = 0.02 * float(np.max(np.abs(y)))
    peaks_idx, props = find_peaks(y, distance=3, prominence=prominence)
    if len(peaks_idx) == 0:
        return np.array([], dtype=np.float64)
    # Sort by height descending, keep top k, then sort by x ascending.
    heights = y[peaks_idx]
    order = np.argsort(heights)[::-1][:k]
    top_idx = np.sort(peaks_idx[order])
    return x[top_idx]


def peak_position_error(
    x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, top_k: int,
) -> Tuple[float, float, int]:
    """Match top-K peaks of both spectra nearest-neighbor and report errors."""
    pa = _top_k_peaks(x, y_a, top_k)
    pb = _top_k_peaks(x, y_b, top_k)
    if len(pa) == 0 or len(pb) == 0:
        return 0.0, 0.0, 0
    # For each a-peak, match to nearest b-peak; take the min(len(a),len(b))
    # best matches to avoid spurious far matches inflating the error.
    deltas = []
    for xa in pa:
        deltas.append(float(np.min(np.abs(pb - xa))))
    n = min(len(pa), len(pb))
    deltas_sorted = sorted(deltas)[:n]
    mean_err = float(np.mean(deltas_sorted))
    max_err = float(np.max(deltas_sorted))
    return mean_err, max_err, n


def l3_l2_ratio(
    x: np.ndarray, y: np.ndarray,
    l3_window: Optional[Tuple[float, float]] = None,
    l2_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Integrated-intensity ratio over (L3, L2) windows.

    Default windows split the available x-range in half at the midpoint
    and treat the lower half as L3, the upper as L2. For 3d L2,3-edge
    spectra this gives a reasonable approximation; pass explicit
    windows for non-default cases.
    """
    if l3_window is None or l2_window is None:
        x_min, x_max = float(x.min()), float(x.max())
        x_mid = 0.5 * (x_min + x_max)
        l3_window = l3_window or (x_min, x_mid)
        l2_window = l2_window or (x_mid, x_max)

    def integ(win):
        lo, hi = win
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(y[mask], x[mask]))

    i3 = integ(l3_window)
    i2 = integ(l2_window)
    if i2 == 0.0:
        return float("inf") if i3 != 0.0 else 0.0
    return i3 / i2


# ─────────────────────────────────────────────────────────────
# End-to-end parity driver
# ─────────────────────────────────────────────────────────────


def _argmax_x(x: np.ndarray, y: np.ndarray) -> float:
    return float(x[int(np.argmax(y))])


def compare(
    x_a: np.ndarray, y_a: np.ndarray,
    x_b: np.ndarray, y_b: np.ndarray,
    calctype: str = "xas",
    spacing_ev: float = COMMON_GRID_SPACING_EV,
    peak_align: bool = True,
) -> ParityResult:
    """Compute all four parity metrics with sensible defaults for `calctype`.

    When ``peak_align=True`` (the default), spectrum ``a`` is shifted so
    its argmax coincides with spectrum ``b``'s argmax before
    interpolation onto a common grid. This is necessary for
    cross-implementation comparisons because multitorch / pyctm /
    ttmult_raw all use different energy-zero conventions. The shift is
    reported as ``peak_shift_ev`` so a reader can verify that the
    spectra differ by a rigid shift (expected) rather than a lineshape
    distortion (the real thing parity metrics are supposed to catch).
    """
    peak_shift = 0.0
    x_a_aligned = x_a
    if peak_align:
        peak_shift = _argmax_x(x_b, y_b) - _argmax_x(x_a, y_a)
        x_a_aligned = x_a + peak_shift

    x, ya_c, yb_c = resample_to_common_grid(
        x_a_aligned, y_a, x_b, y_b, spacing_ev=spacing_ev,
    )
    ya_n = normalize_unit_max(ya_c)
    yb_n = normalize_unit_max(yb_c)

    top_k = PEAK_TOP_K_RIXS if calctype == "rixs" else PEAK_TOP_K_LEDGE
    mean_err, max_err, n_matched = peak_position_error(x, ya_n, yb_n, top_k)

    return ParityResult(
        cosine=cosine_similarity(ya_n, yb_n),
        max_abs_diff=max_abs_diff(ya_n, yb_n),
        peak_err_mean_ev=mean_err,
        peak_err_max_ev=max_err,
        l3_l2_ratio=l3_l2_ratio(x, ya_n),
        l3_l2_ratio_ref=l3_l2_ratio(x, yb_n),
        n_peaks_matched=n_matched,
        common_grid_n=int(x.shape[0]),
        common_grid_min=float(x[0]),
        common_grid_max=float(x[-1]),
        peak_shift_ev=peak_shift,
        peak_aligned=peak_align,
    )
