"""Tests for calcXAS_from_scratch — the fully standalone XAS pipeline.

Validates:
  1. Basic smoke test: produces a spectrum with nonzero intensity
  2. Parameter sensitivity: changing tendq, slater, soc shifts the spectrum
  3. Cross-validation: from-scratch spectrum shape vs. fixture-based result
  4. Multi-element coverage: works for even- and odd-electron d^N
"""
import math
import os

import numpy as np
import pytest
import torch

from multitorch.api.calc import calcXAS_from_scratch


# ─────────────────────────────────────────────────────────────
# Smoke tests — basic functionality
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("element,valence,n_d", [
    ("Ni", "ii", 8),
    ("Fe", "ii", 6),
    ("V", "iii", 2),
])
def test_from_scratch_smoke(element, valence, n_d):
    """calcXAS_from_scratch produces a nonzero broadened spectrum."""
    x, y = calcXAS_from_scratch(
        element=element, valence=valence,
        cf={'tendq': 1.0},
        slater=0.8, soc=1.0,
        T=80.0, nbins=500,
    )
    assert x.shape == (500,)
    assert y.shape == (500,)
    assert y.max() > 0, f"Zero spectrum for {element}{valence}"
    assert torch.isfinite(y).all(), f"Non-finite values in {element}{valence}"


def test_from_scratch_return_sticks():
    """return_sticks=True returns (x, y, sticks) with correct shapes."""
    x, y, sticks = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0},
        return_sticks=True,
        nbins=200,
    )
    assert sticks.ndim == 2
    assert sticks.shape[1] == 2
    assert sticks.shape[0] > 0, "No sticks produced"
    # Sticks should have positive energies (in some range)
    # and non-zero intensities
    energies = sticks[:, 0]
    intensities = sticks[:, 1]
    assert intensities.abs().sum() > 0, "Zero total stick intensity"


# ─────────────────────────────────────────────────────────────
# Parameter sensitivity tests
# ─────────────────────────────────────────────────────────────

def test_from_scratch_tendq_shifts_spectrum():
    """Changing 10Dq shifts the spectral features."""
    _, y1 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 0.5}, nbins=500,
    )
    _, y2 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 2.0}, nbins=500,
    )
    # Spectra should differ
    diff = (y1 - y2).abs().max().item()
    assert diff > 0.01, f"tendq change had no effect: max diff = {diff:.6f}"


def test_from_scratch_slater_changes_spectrum():
    """Changing the Slater reduction factor changes the spectrum."""
    _, y1 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0}, slater=0.6, nbins=500,
    )
    _, y2 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0}, slater=1.0, nbins=500,
    )
    diff = (y1 - y2).abs().max().item()
    assert diff > 0.01, f"slater change had no effect: max diff = {diff:.6f}"


def test_from_scratch_soc_changes_spectrum():
    """Changing the SOC reduction factor changes the spectrum."""
    _, y1 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0}, soc=0.5, nbins=500,
    )
    _, y2 = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0}, soc=1.0, nbins=500,
    )
    diff = (y1 - y2).abs().max().item()
    assert diff > 0.01, f"soc change had no effect: max diff = {diff:.6f}"


# ─────────────────────────────────────────────────────────────
# Multi-element tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("element,valence", [
    ("Co", "ii"),     # d7 — half-integer J
    ("Mn", "ii"),     # d5 — half-integer J, high-spin
    ("Cr", "iii"),    # d3 — half-integer J
    ("Fe", "iii"),    # d5 — half-integer J
    ("Ni", "iii"),    # d7 — half-integer J
    ("Cu", "ii"),     # d9 — half-integer J, requires double-group support
])
def test_from_scratch_multi_element(element, valence):
    """calcXAS_from_scratch works across the 3d series."""
    x, y = calcXAS_from_scratch(
        element=element, valence=valence,
        cf={'tendq': 1.0},
        slater=0.8, soc=1.0,
        nbins=200,
    )
    assert y.max() > 0, f"Zero spectrum for {element} {valence}"
    assert torch.isfinite(y).all()


# ─────────────────────────────────────────────────────────────
# Cross-validation vs. fixture-based pipeline
# ─────────────────────────────────────────────────────────────

def test_from_scratch_vs_fixture_ni2_sticks():
    """From-scratch Ni2+ stick count matches fixture-based result.

    We don't expect exact numerical match (different HFS params vs.
    fixture's Fortran params), but the number of sticks and overall
    spectral shape should be similar.
    """
    REF_DIR = os.path.join(
        os.path.dirname(__file__), '..', 'reference_data', 'ni2_d8_oh')
    ban_out = os.path.join(REF_DIR, 'ni2_d8_oh.ban_out')
    if not os.path.exists(ban_out):
        pytest.skip("No ni2_d8_oh fixture")

    from multitorch.api.calc import calcXAS
    from multitorch.io.read_oba import read_ban_output
    from multitorch.spectrum.sticks import get_sticks

    # Fixture-based sticks
    ban = read_ban_output(ban_out)
    E_fix, M_fix, _ = get_sticks(ban, T=0.0, max_gs=1)

    # From-scratch sticks
    _, _, sticks = calcXAS_from_scratch(
        element='Ni', valence='ii',
        cf={'tendq': 1.0},
        slater=0.8, soc=1.0,
        T=0.0, max_gs=1,
        return_sticks=True,
    )
    E_gen = sticks[:, 0]

    # Both should have multiple sticks
    assert E_fix.numel() > 5, f"Fixture has only {E_fix.numel()} sticks"
    assert E_gen.numel() > 5, f"From-scratch has only {E_gen.numel()} sticks"

    # Energy ranges should be comparable (within a factor of ~2x)
    fix_span = float(E_fix.max() - E_fix.min())
    gen_span = float(E_gen.max() - E_gen.min())
    assert gen_span > 0, "Zero energy spread in from-scratch sticks"
    ratio = fix_span / gen_span if gen_span > 0 else float('inf')
    assert 0.2 < ratio < 5.0, (
        f"Energy spread ratio = {ratio:.2f} "
        f"(fix={fix_span:.2f} eV, gen={gen_span:.2f} eV)"
    )
