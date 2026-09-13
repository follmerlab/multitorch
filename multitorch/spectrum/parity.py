"""
Spectral parity on the union of two energy windows (WP-S S7).

Both spectra are resampled onto the union of their x-ranges, with zero
intensity outside each one's own range, so intensity that one spectrum puts
where the other has none lowers the cosine instead of being cropped away. No
peak alignment is done: pass ``shift_a`` only for a *known* energy-zero
convention difference (e.g. a fixture's configuration-average offset). The
fraction of each spectrum's intensity inside the overlap of the two ranges is
reported so a narrow comparison window cannot hide missing intensity (audit
§7.1: a 0.978 cosine had been computed after an 830 eV argmax shift on a
window holding 66% of the intensity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SpectralParity:
    cosine: float
    max_abs_diff: float        # after normalising both to unit maximum
    fraction_inside_a: float   # of ∫|y_a| within the overlap of the two x-ranges
    fraction_inside_b: float
    area_ratio: float          # ∫y_a / ∫y_b
    l3_l2_a: Optional[float] = None
    l3_l2_b: Optional[float] = None


def _as_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def spectral_parity(x_a, y_a, x_b, y_b, *, shift_a: float = 0.0,
                    split: Optional[float] = None) -> SpectralParity:
    """Compare spectrum a (shifted by ``shift_a`` eV) with spectrum b on the union window.

    ``split``: energy (on b's scale) separating L3 from L2; when given, the
    integrated L3/L2 intensity ratio of each spectrum is reported.
    """
    xa, ya, xb, yb = _as_np(x_a) + shift_a, _as_np(y_a), _as_np(x_b), _as_np(y_b)
    step = min(np.min(np.diff(xa)), np.min(np.diff(xb)))
    lo, hi = min(xa[0], xb[0]), max(xa[-1], xb[-1])
    x = np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
    a = np.interp(x, xa, ya, left=0.0, right=0.0)
    b = np.interp(x, xb, yb, left=0.0, right=0.0)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
    max_abs = float(np.max(np.abs(a / np.max(np.abs(a)) - b / np.max(np.abs(b))))) if na > 0 and nb > 0 else 1.0

    inside = (x >= max(xa[0], xb[0])) & (x <= min(xa[-1], xb[-1]))
    fa = float(np.abs(a[inside]).sum() / np.abs(a).sum()) if na > 0 else 0.0
    fb = float(np.abs(b[inside]).sum() / np.abs(b).sum()) if nb > 0 else 0.0

    l3l2 = (None, None)
    if split is not None:
        def ratio(y):
            hi_part = np.trapezoid(y[x > split], x[x > split])
            return float(np.trapezoid(y[x <= split], x[x <= split]) / hi_part) if hi_part else float("inf")
        l3l2 = (ratio(a), ratio(b))

    return SpectralParity(
        cosine=cosine, max_abs_diff=max_abs,
        fraction_inside_a=fa, fraction_inside_b=fb,
        area_ratio=float(np.trapezoid(a, x) / np.trapezoid(b, x)),
        l3_l2_a=l3l2[0], l3_l2_b=l3l2[1],
    )
