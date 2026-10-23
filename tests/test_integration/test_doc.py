"""
Tests for multitorch.api.calc.calcDOC — ground-state configuration weights.

Two modes are tested:
  1. Bootstrap: reads config weights from .ban_out header metadata.
  2. Phase 5: computes weights from assembled Hamiltonian eigenvectors.
"""
from __future__ import annotations

import collections
from pathlib import Path

from unittest import mock

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


# ─────────────────────────────────────────────────────────────
# Residual 34 and the per-triad reference energy
# ─────────────────────────────────────────────────────────────

def _fortran_band_weights(ban_out: Path):
    """ttban's own per-BAND ground-state weights, read from the file as text.

    Read with a local reader rather than through :func:`read_ban_output`, so
    that the comparison in
    ``test_per_triad_weights_reproduce_the_fortran_per_band_weights`` is
    file-against-computation and not our parser against our assembler.

    ttban writes each band's triad as the non-blank line before its "Ground
    state energy", then its weights. The half-integer files add a fourth
    token, the PRMULT channel-copy index, so only the first three are the
    triad.
    """
    out, prev = [], ""
    for line in ban_out.read_text().splitlines():
        if "Ground state energy Eg0=" in line:
            out.append([prev.split()[:3], None])
        elif "Weight of configurations" in line:
            out[-1][1] = [float(x) for x in line.split(":")[-1].split()]
        if line.strip():
            prev = line.strip()
    return out


def test_per_triad_weights_reproduce_the_fortran_per_band_weights():
    """The Fortran oracle for calcDOC's per-triad breakdown, which had none.

    ttban prints "Weight of configurations ... in the ground state" once per
    BAND; that is the same quantity ``per_triad[i]['config_weights']`` claims
    to be, and nothing in the suite compared them. **Measured before the fix,
    3 of Ni(II) D4h's 5 ground irreps came back 0.000000 against ttban's
    0.91158, 0.91552 and 0.90889** -- an absolute error of up to 0.92 on a
    documented return value.

    The cause was not the physics but a floor. The ratio
    ``Sum cw*boltz / Sum boltz`` is invariant under rescaling ``boltz``, so
    referencing it to the global ground state was harmless in exact
    arithmetic; the ``max(z_triad, 1e-30)`` guard broke the invariance, because
    a triad 1.26 eV up has ``z = 4e-80`` at 80 K and the floor replaced it.
    """
    from multitorch.api.calc import fixture_defaults

    d = fixture_defaults("Ni", "ii", "d4h")
    doc = calcDOC("Ni", "ii", "d4h", "l", cf=d["cf"], delta=d["delta"],
                  u=d["u"], lmct=d["lmct"], T=80.0)
    fortran = {tri[0]: w for tri, w in
               _fortran_band_weights(REFDATA / "nid8ct" / "nid8ct.ban_out")}
    assert len(fortran) == 5, sorted(fortran)

    ours = {}
    for t in doc["per_triad"]:
        ours.setdefault(t["sym"].split()[0], []).append(t["config_weights"])
    assert set(ours) == set(fortran), (sorted(ours), sorted(fortran))

    for gs, rows in ours.items():
        # A ground-state property cannot depend on which final state the triad
        # reaches, so every listing of one irrep must agree exactly. This is
        # the invariance the floor broke: it scaled each listing by that
        # triad's own population.
        for row in rows[1:]:
            assert row == rows[0], (gs, rows)
        for c, (mine, theirs) in enumerate(zip(rows[0], fortran[gs])):
            assert mine == pytest.approx(theirs, abs=1e-5), (gs, c, mine, theirs)


def test_the_aggregate_counts_ground_states_not_triad_listings():
    """Residual 34, as an invariance: duplicating a listing must change nothing.

    ttban writes one BAND per triad *listing*, so a ground irrep reached by
    several final states -- or carrying two invariant channels, the repeated
    Gamma8 (x) T1u (x) Gamma8 coupling -- is printed several times with
    identical weights. nid8ct lists '1+' five times and '0+' twice, so
    averaging per listing makes '1+' five times as important as it is.

    Asserted by duplicating a band rather than by comparing against a second
    implementation. A duplicate *listing* of one ground state is physically a
    no-op for a ground-state property, so the aggregate must be invariant
    under it; the defect is exactly a violation of that invariance. The
    measured bias on the bundled fixtures is small (<= 4.7e-06, and exactly 0
    on three of the nine, because one irrep dominates the Boltzmann pool and
    its multiplicity cancels top and bottom), which is why this is tested as a
    structural property and not by a number moving.
    """
    import dataclasses

    from multitorch.api.calc import _calcDOC_bootstrap
    from multitorch.io.read_oba import read_ban_output

    path = str(REFDATA / "nid8ct" / "nid8ct.ban_out")
    base = _calcDOC_bootstrap(path, T=80.0)

    bo = read_ban_output(path)
    counts = collections.Counter(bm.triad_sym.split()[0] for bm in bo.band_metadata)
    assert counts["1+"] == 5 and counts["0+"] == 2, counts

    # Duplicate the most-listed ground irrep's band another three times.
    extra = next(bm for bm in bo.band_metadata if bm.triad_sym.split()[0] == "1+")
    bo.band_metadata.extend(dataclasses.replace(extra) for _ in range(3))
    with mock.patch("multitorch.io.read_oba.read_ban_output", return_value=bo):
        skewed = _calcDOC_bootstrap(path, T=80.0)

    assert skewed["config_weights"] == pytest.approx(base["config_weights"], abs=1e-12)


def test_every_band_carries_its_triad_symmetries():
    """``read_ban_output`` populated ``triad_sym`` for one band per file.

    ttban writes the ``TRIADS`` keyword once per BAND and then lists each
    triad as a bare line, so the keyword-triggered branch reached only the
    first -- 1 of 13 bands for nid8ct. Residual 34's fix needs the ground
    irrep of *every* band to count ground states rather than listings.

    The oracle is the paired ``.ban``, parsed by a different parser: the two
    files are written independently by the Fortran chain and must agree on
    the triads and their order. Across the ten bundled pairs that is 95 of 95
    bands, and before the fix it was 10 of 95.
    """
    from multitorch.io.read_ban import read_ban
    from multitorch.io.read_oba import read_ban_output

    checked = 0
    for ban_path in sorted(REFDATA.glob("*/*.ban")):
        out_path = ban_path.with_suffix(".ban_out")
        if not out_path.exists():
            continue
        triads = read_ban(str(ban_path)).triads
        bands = read_ban_output(str(out_path)).band_metadata
        assert len(bands) == len(triads), (ban_path.name, len(bands), len(triads))
        for bm, triad in zip(bands, triads):
            assert tuple(bm.triad_sym.split()[:3]) == tuple(triad), (
                ban_path.name, bm.triad_sym, triad)
        checked += len(triads)
    assert checked == 95, checked
