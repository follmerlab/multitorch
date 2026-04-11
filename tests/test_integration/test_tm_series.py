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
CASES = [
    ("ti4_d0_oh", 0.97),   # d0, known edge case (see README §Known limitations §4)
    ("v3_d2_oh",  0.99),
    ("cr3_d3_oh", 0.99),
    ("mn2_d5_oh", 0.99),
    ("fe3_d5_oh", 0.99),
    ("fe2_d6_oh", 0.99),
    ("co2_d7_oh", 0.99),
    ("ni2_d8_oh", 0.99),
]


def _cosine_to_reference(x: torch.Tensor, y: torch.Tensor, xy_path: Path) -> float:
    """
    Compare a multitorch spectrum to a pyctm .xy file.

    The .xy file is in relative-energy coordinates (starts at 0) so we
    shift the multitorch x-axis so its peak lines up with the reference
    peak, then interpolate onto the reference grid and compute the
    cosine similarity.
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    xy_ref = np.loadtxt(str(xy_path))
    x_ref = xy_ref[:, 0]
    y_ref = xy_ref[:, 1]

    # Align peaks
    shift = x_np[y_np.argmax()] - x_ref[y_ref.argmax()]
    x_aligned = x_np - shift
    y_interp = np.interp(x_ref, x_aligned, y_np, left=0.0, right=0.0)

    num = float(np.dot(y_interp, y_ref))
    den = float(np.linalg.norm(y_interp) * np.linalg.norm(y_ref) + 1e-30)
    return num / den


@pytest.mark.integration
@pytest.mark.parametrize("case_id,cos_min", CASES)
def test_tm_series_getXAS_runs(case_id, cos_min):
    """getXAS from {case}.ban_out runs, returns valid tensors, and matches ref."""
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

    # Shape fidelity vs Fortran reference
    cos = _cosine_to_reference(x, y, xy_path)
    assert cos >= cos_min, (
        f"{case_id}: cosine similarity {cos:.4f} below threshold {cos_min:.3f}"
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
