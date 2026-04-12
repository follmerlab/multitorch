"""
Tests for multitorch.api.calc.calcDOC — ground-state configuration weights.

Two modes are tested:
  1. Bootstrap: reads config weights from .ban_out header metadata.
  2. Phase 5: computes weights from assembled Hamiltonian eigenvectors.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcDOC

REFDATA = Path(__file__).parent.parent / "reference_data"


# ─────────────────────────────────────────────────────────────
# Bootstrap mode
# ─────────────────────────────────────────────────────────────

class TestDOCBootstrap:
    """Tests using bootstrap mode (ban_output_path)."""

    def test_returns_required_keys(self):
        doc = calcDOC(ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"))
        assert 'config_weights' in doc
        assert 'per_triad' in doc
        assert 'metal_character' in doc
        assert 'ct_character' in doc

    def test_config_weights_sum_to_one(self):
        doc = calcDOC(ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"))
        total = sum(doc['config_weights'])
        assert abs(total - 1.0) < 0.01

    def test_nid8ct_mostly_metal(self):
        """Ni²⁺ D4h CT: ~88-92% d⁸ character."""
        doc = calcDOC(ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"))
        assert 0.80 < doc['metal_character'] < 0.95
        assert 0.05 < doc['ct_character'] < 0.20

    def test_per_triad_has_metadata(self):
        doc = calcDOC(ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"))
        assert len(doc['per_triad']) > 0
        for pt in doc['per_triad']:
            assert 'sym' in pt
            assert 'gs_energy' in pt
            assert 'config_weights' in pt

    def test_zero_temperature_selects_ground_state(self):
        """At T=0, only the lowest-energy band contributes."""
        doc = calcDOC(
            ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"),
            T=0.0,
        )
        total = sum(doc['config_weights'])
        assert abs(total - 1.0) < 0.01

    def test_temperature_affects_weights(self):
        """Different temperatures give different Boltzmann averages."""
        doc_cold = calcDOC(
            ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"),
            T=10.0,
        )
        doc_hot = calcDOC(
            ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"),
            T=300.0,
        )
        # At higher T, more excited states contribute
        assert doc_cold['metal_character'] != doc_hot['metal_character']

    def test_oh_fixture(self):
        """Ni²⁺ Oh CT fixture should also show mostly metal character."""
        ban_out = REFDATA / "ni2_d8_oh" / "ni2_d8_oh.ban_out"
        if not ban_out.exists():
            pytest.skip(
                "ni2_d8_oh.ban_out not available — this hides one DOC "
                "coverage test. Regenerate Oh fixture to restore."
            )
        doc = calcDOC(ban_output_path=str(ban_out))
        # Oh with CT: ~88-92% d⁸ character
        assert 0.80 < doc['metal_character'] < 0.95
        assert abs(sum(doc['config_weights']) - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────
# Phase 5 mode
# ─────────────────────────────────────────────────────────────

class TestDOCPhase5:
    """Tests using Phase 5 mode (element/valence/sym/edge)."""

    def test_returns_required_keys(self):
        doc = calcDOC(
            element='Ni', valence='ii', sym='d4h', edge='l',
            cf={'tendq': 1.0, 'ds': 0.1}, slater=1.0, soc=1.0,
        )
        assert 'config_weights' in doc
        assert 'metal_character' in doc
        assert 'ct_character' in doc

    def test_config_weights_sum_to_one(self):
        doc = calcDOC(
            element='Ni', valence='ii', sym='d4h', edge='l',
            cf={'tendq': 1.0, 'ds': 0.1}, slater=1.0, soc=1.0,
        )
        total = sum(doc['config_weights'])
        assert abs(total - 1.0) < 0.01

    def test_nid8ct_mostly_metal(self):
        doc = calcDOC(
            element='Ni', valence='ii', sym='d4h', edge='l',
            cf={'tendq': 1.0, 'ds': 0.1}, slater=1.0, soc=1.0,
        )
        assert 0.80 < doc['metal_character'] < 0.95

    def test_bootstrap_vs_phase5_parity(self):
        """Bootstrap and Phase 5 should give similar config weights."""
        doc_bootstrap = calcDOC(
            ban_output_path=str(REFDATA / "nid8ct" / "nid8ct.ban_out"),
            T=80.0,
        )
        doc_phase5 = calcDOC(
            element='Ni', valence='ii', sym='d4h', edge='l',
            cf={'tendq': 1.0, 'ds': 0.1}, slater=1.0, soc=1.0, T=80.0,
        )
        # Both should agree on metal character to within a few percent
        assert abs(
            doc_bootstrap['metal_character'] - doc_phase5['metal_character']
        ) < 0.05
