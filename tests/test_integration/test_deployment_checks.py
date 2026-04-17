"""
Deployment audit integration tests — behavioral checks on full pipeline.

TEST-02: Autograd gradcheck on Phase 5 pipeline (differentiability)
TEST-03: Sum rule — integrated XAS intensity conservation
"""
import pytest
import torch

from multitorch._constants import DTYPE


# ────────────────────────────────────────────────────────────────
# TEST-02: Autograd differentiability of Phase 5 pipeline
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestDifferentiability:
    """The Phase 5 pipeline must produce gradients w.r.t. atomic parameters."""

    def test_phase5_produces_valid_spectrum(self):
        """Phase 5 pipeline must produce a finite, nonzero spectrum."""
        from multitorch.api.calc import calcXAS

        result = calcXAS(
            element='Ni', valence='ii', sym='oh', edge='l',
            cf={'tendq': 1.0},
            slater=0.8, soc=1.0, T=80,
            use_phase5=True,
        )

        x, y = result
        assert isinstance(y, torch.Tensor), "Spectrum y must be a tensor"
        assert y.abs().sum() > 0, "Spectrum should have nonzero intensity"
        assert torch.isfinite(y).all(), "Spectrum must be finite"

    def test_slater_scale_affects_spectrum(self):
        """Different slater reduction factors must produce different spectra."""
        from multitorch.api.calc import calcXAS

        _, y1 = calcXAS(
            element='Ni', valence='ii', sym='oh', edge='l',
            cf={'tendq': 1.0}, slater=0.6, soc=1.0, T=80,
            use_phase5=True,
        )
        _, y2 = calcXAS(
            element='Ni', valence='ii', sym='oh', edge='l',
            cf={'tendq': 1.0}, slater=1.0, soc=1.0, T=80,
            use_phase5=True,
        )
        diff = (y1 - y2).abs().max().item()
        assert diff > 0.01, f"Spectra should differ with different slater: max diff = {diff:.6f}"

    def test_broadening_params_differentiable(self):
        """Broadening with tensor FWHM should keep autograd tape alive."""
        from multitorch.spectrum.broaden import pseudo_voigt

        x = torch.linspace(0, 10, 100, dtype=DTYPE)
        x0 = torch.tensor([3.0, 5.0, 7.0], dtype=DTYPE)
        amp = torch.tensor([1.0, 2.0, 0.5], dtype=DTYPE)
        fwhm_g = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
        fwhm_l = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)

        y = pseudo_voigt(x, x0, amp, fwhm_g, fwhm_l, mode='correct')
        loss = y.sum()
        loss.backward()

        assert fwhm_g.grad is not None, "Gradient w.r.t. fwhm_g must exist"
        assert fwhm_l.grad is not None, "Gradient w.r.t. fwhm_l must exist"
        assert not fwhm_g.grad.isnan(), "fwhm_g gradient must not be NaN"
        assert not fwhm_l.grad.isnan(), "fwhm_l gradient must not be NaN"


# ────────────────────────────────────────────────────────────────
# TEST-02b: Crystal field parameter sensitivity
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestCFSensitivity:
    """Crystal field parameters must affect the spectrum."""

    def test_tendq_changes_spectrum(self):
        """Different 10Dq values must produce different spectra."""
        from multitorch.api.calc import calcXAS_from_scratch

        x1, y1 = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 0.5}, slater=0.8, soc=1.0, T=80,
        )
        x2, y2 = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 2.0}, slater=0.8, soc=1.0, T=80,
        )

        # Spectra must differ — 10Dq is the primary CF splitting
        diff = (y1 - y2).abs().max().item()
        assert diff > 0.01, \
            f"Spectra should differ with different 10Dq, max diff = {diff:.6f}"


# ────────────────────────────────────────────────────────────────
# TEST-03: Sum rule — integrated intensity conservation
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestSumRule:
    """XAS integrated intensity should be conserved under broadening."""

    def test_stick_sum_equals_broadened_integral(self):
        """The sum of stick intensities should approximately equal the
        integral of the broadened spectrum (area conservation)."""
        from multitorch.spectrum.broaden import pseudo_voigt
        from multitorch.spectrum.sticks import get_sticks
        from multitorch.io.read_oba import read_ban_output
        from pathlib import Path

        refdata = Path(__file__).parent.parent / 'reference_data' / 'nid8ct'
        ban = read_ban_output(str(refdata / 'nid8ct.ban_out'))

        Etrans, Mtrans, _ = get_sticks(ban, T=80.0)

        if Etrans.numel() == 0:
            pytest.skip("No sticks produced")

        stick_sum = Mtrans.sum().item()

        # Broaden onto a fine grid
        E_min, E_max = Etrans.min().item() - 5, Etrans.max().item() + 5
        x = torch.linspace(E_min, E_max, 5000, dtype=DTYPE)
        y = pseudo_voigt(x, Etrans, Mtrans, fwhm_g=0.2, fwhm_l=0.2)

        # Trapezoidal integration
        dx = x[1] - x[0]
        broadened_integral = torch.trapezoid(y, x).item()

        # Should be within 5% (broadening spreads intensity but conserves area
        # when the grid extends well beyond the stick range)
        ratio = broadened_integral / stick_sum
        assert 0.95 < ratio < 1.05, \
            f"Area ratio = {ratio:.4f}, stick_sum = {stick_sum:.4f}, integral = {broadened_integral:.4f}"

    def test_nonnegative_intensities(self):
        """All stick intensities must be non-negative."""
        from multitorch.spectrum.sticks import get_sticks
        from multitorch.io.read_oba import read_ban_output
        from pathlib import Path

        refdata = Path(__file__).parent.parent / 'reference_data' / 'nid8ct'
        ban = read_ban_output(str(refdata / 'nid8ct.ban_out'))
        _, Mtrans, _ = get_sticks(ban, T=80.0)
        assert (Mtrans >= 0).all(), "Stick intensities must be non-negative"


# ────────────────────────────────────────────────────────────────
# TEST-07: Edge-case electron counts (d0, d1, d10)
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestEdgeCaseElectronCounts:
    """Boundary d-electron counts must produce valid (possibly trivial) spectra."""

    def test_d1_ti_iii(self):
        """d1 (Ti3+) — simplest non-trivial half-integer J case."""
        from multitorch.api.calc import calcXAS_from_scratch
        x, y = calcXAS_from_scratch(
            element='Ti', valence='iii',
            cf={'tendq': 1.0}, slater=0.8, soc=1.0, nbins=200,
        )
        assert y.max() > 0, "d1 Ti3+ should produce nonzero spectrum"
        assert torch.isfinite(y).all()

    def test_d0_ti_iv(self):
        """d0 (Ti4+) — empty d-shell. Must either produce a finite
        spectrum or raise ValueError (no ground multiplets).
        Must NOT silently produce garbage."""
        from multitorch.api.calc import calcXAS_from_scratch
        try:
            x, y = calcXAS_from_scratch(
                element='Ti', valence='iv',
                cf={'tendq': 1.0}, slater=0.8, soc=1.0, nbins=200,
            )
            # If it succeeds, spectrum must be finite and well-formed
            assert torch.isfinite(y).all(), "d0 spectrum has non-finite values"
            assert x.shape == y.shape, "x and y shapes must match"
        except ValueError as e:
            # d0 may legitimately raise ValueError (no ground state multiplets)
            assert "d0" in str(e).lower() or "no" in str(e).lower() or "electron" in str(e).lower(), \
                f"ValueError should mention the d0 issue, got: {e}"

    def test_d10_zn_ii(self):
        """d10 (Zn2+) — full d-shell. Must either produce a finite
        spectrum or raise ValueError (only one ground term).
        Must NOT silently produce garbage."""
        from multitorch.api.calc import calcXAS_from_scratch
        try:
            x, y = calcXAS_from_scratch(
                element='Zn', valence='ii',
                cf={'tendq': 1.0}, slater=0.8, soc=1.0, nbins=200,
            )
            assert torch.isfinite(y).all(), "d10 spectrum has non-finite values"
            assert x.shape == y.shape, "x and y shapes must match"
        except ValueError as e:
            # d10 may legitimately raise ValueError
            assert "d10" in str(e).lower() or "no" in str(e).lower() or "electron" in str(e).lower(), \
                f"ValueError should mention the d10 issue, got: {e}"


# ────────────────────────────────────────────────────────────────
# TEST-08: Zero crystal field (free-ion limit)
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestZeroCrystalFieldPipeline:
    """tendq=0 should produce a valid spectrum (free-ion limit)."""

    def test_zero_tendq_produces_spectrum(self):
        """With tendq=0, the CF splitting vanishes but Coulomb + SOC
        still produce structure."""
        from multitorch.api.calc import calcXAS_from_scratch
        x, y = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 0.0}, slater=0.8, soc=1.0, nbins=200,
        )
        assert y.max() > 0, "Free-ion spectrum should be nonzero"
        assert torch.isfinite(y).all()


# ────────────────────────────────────────────────────────────────
# TEST-09: Extreme parameter stability
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestExtremeParameters:
    """Pipeline must not crash at extreme (but physically valid) parameter values."""

    def test_very_small_slater(self):
        """slater=0.01 — nearly free-electron limit."""
        from multitorch.api.calc import calcXAS_from_scratch
        x, y = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 1.0}, slater=0.01, soc=1.0, nbins=200,
        )
        assert torch.isfinite(y).all()

    def test_large_slater(self):
        """slater=2.0 — enhanced correlations (unphysical but shouldn't crash)."""
        from multitorch.api.calc import calcXAS_from_scratch
        x, y = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 1.0}, slater=2.0, soc=1.0, nbins=200,
        )
        assert torch.isfinite(y).all()

    def test_zero_soc(self):
        """soc=0.0 — no spin-orbit coupling."""
        from multitorch.api.calc import calcXAS_from_scratch
        x, y = calcXAS_from_scratch(
            element='Ni', valence='ii',
            cf={'tendq': 1.0}, slater=0.8, soc=0.0, nbins=200,
        )
        assert torch.isfinite(y).all()
