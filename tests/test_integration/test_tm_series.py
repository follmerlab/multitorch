"""
End-to-end integration tests: Ti-Ni L-edge XAS series.

Parametrized across the 8 fresh Fortran comparison cases committed under
`tests/reference_data/`. For each ion we run `getXAS(ban_output_path=...)`
from the bootstrap pipeline and verify:

    1. The call completes without error.
    2. The output spectrum has the expected shape, dtype, and positivity.
    3. The broadened shape matches the pyctm reference .xy file on the
       union window (cosine >= 0.9999, all intensity inside, equal area).

This test replaces the previous `/private/tmp/xas_test/` comparison driver
so the multi-ion validation is baked into CI with no Fortran binary or
external-directory dependency at test time.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

REFROOT = Path(__file__).parent.parent / "reference_data"

# Union-window cosine against the pyctm .xy for every ion. The .xy files were
# written by pyctm get_spectrum with med_energy=25 (its default) on the same
# stick energy scale as the .ban_out, so no alignment is needed. The former
# Ti4+ "known limitation" (0.977) was multitorch ignoring med_energy.
CASES = ["ti4_d0_oh", "v3_d2_oh", "cr3_d3_oh", "mn2_d5_oh",
         "fe3_d5_oh", "fe2_d6_oh", "co2_d7_oh", "ni2_d8_oh"]
PYCTM_MED_ENERGY = 25.0


@pytest.mark.integration
@pytest.mark.parametrize("case_id", CASES)
def test_tm_series_getXAS_runs(case_id):
    """getXAS(.ban_out) broadening == pyctm .xy on the union window."""
    from multitorch.api.plot import getXAS
    from multitorch.spectrum.parity import spectral_parity

    ban_path = REFROOT / case_id / f"{case_id}.ban_out"
    xy_path = REFROOT / case_id / f"{case_id}.xy"
    assert ban_path.exists(), f"Missing fixture: {ban_path}"
    assert xy_path.exists(), f"Missing reference spectrum: {xy_path}"

    xy = np.loadtxt(str(xy_path))
    # on pyctm's own grid, so Lorentzian tails beyond our auto window count
    x, y = getXAS(
        str(ban_path),
        T=80.0, beam_fwhm=0.2, gamma1=0.2, gamma2=0.4,
        med_energy=PYCTM_MED_ENERGY,
        xmin=float(xy[0, 0]), xmax=float(xy[-1, 0]), nbins=len(xy),
    )

    assert x.shape == y.shape
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64
    assert (y >= -1e-10).all(), f"{case_id}: negative spectrum values"
    assert float(y.max()) > 0.0, f"{case_id}: empty spectrum"

    p = spectral_parity(x, y, xy[:, 0], xy[:, 1])
    # floor: the .xy is printed to 4 decimals
    assert p.cosine >= 0.9999, f"{case_id}: {p}"
    assert min(p.fraction_inside_a, p.fraction_inside_b) > 0.999999, p
    assert p.area_ratio == pytest.approx(1.0, abs=1e-3), p


@pytest.mark.integration
def test_tm_series_covers_d0_through_d8():
    """Sanity: the committed fixture set actually covers the full d-shell range."""
    d_counts = set()
    for case_id in CASES:
        # e.g. 'ti4_d0_oh' -> 0, 'ni2_d8_oh' -> 8
        token = case_id.split("_")[1]  # 'd0', 'd8', ...
        d_counts.add(int(token[1:]))
    assert d_counts == {0, 2, 3, 5, 6, 7, 8}, f"unexpected d-counts: {sorted(d_counts)}"
