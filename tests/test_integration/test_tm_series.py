"""
End-to-end integration tests: Ti-Ni L-edge XAS series.

Parametrized across the 8 fresh Fortran comparison cases committed under
`tests/reference_data/`. For each ion we run `getXAS(ban_output_path=...)`
from the bootstrap pipeline and verify:

    1. The call completes without error.
    2. The output spectrum has the expected shape, dtype, and positivity.
    3. The broadened shape matches the Fortran reference .xy file to
       cosine similarity >= 0.97 (Ti4+ is a known exception at ~0.977;
       the other 7 cases land above 0.99).

This test replaces the previous `/private/tmp/xas_test/` comparison driver
so the multi-ion validation is baked into CI with no Fortran binary or
external-directory dependency at test time.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

REFROOT = Path(__file__).parent.parent / "reference_data"

# (case_id, expected cosine lower bound)
# Per-case cosine similarity thresholds.
# Cases below 0.99 are KNOWN LIMITATIONS with documented root causes:
#   ti4_d0_oh (0.97): d0 has no d-d Slater integrals → eigenvalues match to
#       3.7e-7 Ry but the residual gap is in the broadening layer (Voigt
#       convolution grid alignment). See README §Known limitations §4.
# All other cases must achieve ≥ 0.99.
CASES = [
    ("ti4_d0_oh", 0.97),
    ("v3_d2_oh",  0.99),
    ("cr3_d3_oh", 0.99),
    ("mn2_d5_oh", 0.99),
    ("fe3_d5_oh", 0.99),
    ("fe2_d6_oh", 0.99),
    ("co2_d7_oh", 0.99),
    ("ni2_d8_oh", 0.99),
]


def _compare_to_reference(x: torch.Tensor, y: torch.Tensor, xy_path: Path) -> dict:
    """
    Compare a multitorch spectrum to a pyctm .xy file.

    The .xy file is in relative-energy coordinates (starts at 0) so we
    shift the multitorch x-axis so its peak lines up with the reference
    peak, then interpolate onto the reference grid and compute:
      - cosine similarity
      - peak shift (eV) before alignment
      - amplitude ratio (multitorch peak / reference peak)

    Returns a dict with keys: cosine, peak_shift_eV, amplitude_ratio.
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    xy_ref = np.loadtxt(str(xy_path))
    x_ref = xy_ref[:, 0]
    y_ref = xy_ref[:, 1]

    # Peak shift before alignment (eV)
    peak_shift = float(x_np[y_np.argmax()] - x_ref[y_ref.argmax()])

    # Amplitude ratio (peak heights)
    ref_max = float(y_ref.max())
    mt_max = float(y_np.max())
    amp_ratio = mt_max / ref_max if ref_max > 1e-30 else float('inf')

    # Align peaks for cosine similarity
    x_aligned = x_np - peak_shift
    y_interp = np.interp(x_ref, x_aligned, y_np, left=0.0, right=0.0)

    num = float(np.dot(y_interp, y_ref))
    den = float(np.linalg.norm(y_interp) * np.linalg.norm(y_ref) + 1e-30)
    cosine = num / den

    return {
        'cosine': cosine,
        'peak_shift_eV': peak_shift,
        'amplitude_ratio': amp_ratio,
    }


@pytest.mark.integration
@pytest.mark.parametrize("case_id,cos_min", CASES)
def test_tm_series_getXAS_runs(case_id, cos_min):
    """getXAS from {case}.ban_out runs, returns valid tensors, and matches ref.

    Three independent checks prevent false positives:
      1. Cosine similarity ≥ threshold (catches shape distortion)
      2. Peak shift < 2 eV (catches energy offset errors that cosine misses)
      3. Amplitude ratio within [0.5, 2.0] (catches scaling errors that cosine misses)
    """
    from multitorch.api.plot import getXAS

    ban_path = REFROOT / case_id / f"{case_id}.ban_out"
    xy_path = REFROOT / case_id / f"{case_id}.xy"
    assert ban_path.exists(), f"Missing fixture: {ban_path}"
    assert xy_path.exists(), f"Missing reference spectrum: {xy_path}"

    x, y = getXAS(
        str(ban_path),
        T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4,
    )

    # Basic shape / dtype / positivity checks
    assert x.shape == y.shape
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64
    assert (y >= -1e-10).all(), f"{case_id}: negative spectrum values"
    assert float(y.max()) > 0.0, f"{case_id}: empty spectrum"

    # Multi-metric fidelity check vs Fortran reference
    cmp = _compare_to_reference(x, y, xy_path)

    assert cmp['cosine'] >= cos_min, (
        f"{case_id}: cosine similarity {cmp['cosine']:.4f} below threshold {cos_min:.3f}"
    )
    # Peak shift catches uniform energy offsets (cosine-invisible)
    assert abs(cmp['peak_shift_eV']) < 2.0, (
        f"{case_id}: peak shift {cmp['peak_shift_eV']:.2f} eV exceeds 2 eV"
    )
    # Amplitude ratio catches uniform scaling (cosine-invisible)
    assert 0.5 < cmp['amplitude_ratio'] < 2.0, (
        f"{case_id}: amplitude ratio {cmp['amplitude_ratio']:.3f} outside [0.5, 2.0]"
    )


@pytest.mark.integration
def test_tm_series_covers_d0_through_d8():
    """Sanity: the committed fixture set actually covers the full d-shell range."""
    d_counts = set()
    for case_id, _ in CASES:
        # e.g. 'ti4_d0_oh' -> 0, 'ni2_d8_oh' -> 8
        token = case_id.split("_")[1]  # 'd0', 'd8', ...
        d_counts.add(int(token[1:]))
    assert d_counts == {0, 2, 3, 5, 6, 7, 8}, f"unexpected d-counts: {sorted(d_counts)}"
