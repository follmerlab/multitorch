"""Dipole sum rule: the integrated L2,3 intensity counts 3d holes.

Oracle (analytic, closure over the complete final manifold): with a full 2p
shell in the ground state, Σ_f |⟨f|D|g⟩|² = c · ⟨g| n_h(3d) |g⟩ for every ground
component g, independent of Coulomb, spin-orbit, crystal field and
hybridisation (the ligand hole is not dipole-coupled to 2p). With pyctm's
unnormalised Boltzmann weights (decision D-1) at ``max_gs=1`` the stick sum is
therefore

    ΣI = 0.2 · ⟨n_h⟩ · (number of ground components counted)

with ⟨n_h⟩ the configuration-weighted metal hole count
(:func:`multitorch.analysis.analyze_ground_state`). The Fortran Cr³⁺
``.ban_out`` obeys it too (ΣI 5.3995 vs 5.3994 predicted).

It caught residual 14 (Cr³⁺ fixture, ratio 4.008 at the template and 4.155 at
10Dq 0.1: the second PRMULT copy of S1+→S1− was never assembled; fixed) and it
is what flagged the legacy half-integer dipole (from-scratch ratios 4.5-10.2,
now integers: WP-B2a-3), and it is the oracle that closed residual 15.
Until 2026-10-18 ``max_gs`` counted exactly equal floats, so the Oh d5 ions
counted 2 of their 6 near-degenerate ground components and this test xfailed
for them on the fixture and from-scratch paths alike. ``get_sticks*`` now
pools by level (:data:`~multitorch.spectrum.sticks.LEVEL_WINDOW_EV`) and
weights each member by the level's energy, which is what keeps the ratio an
exact integer instead of one scattered by the intra-level weight spread.
"""
from __future__ import annotations

import functools

import pytest

from multitorch.analysis import analyze_ground_state
from multitorch.analysis.ground_state import configuration_label
from multitorch.api.calc import calcXAS_cached, preload_fixture, preload_from_scratch

T = 80.0

FIXTURES = [("Ti", "iv"), ("V", "iii"), ("Cr", "iii"), ("Mn", "ii"), ("Fe", "ii"), ("Fe", "iii"), ("Co", "ii"), ("Ni", "ii")]
REGIMES = {
    "template": {},
    "ionic_10dq0.1": dict(cf={"tendq": 0.1}, delta=100.0, lmct=0.0),
}


@functools.lru_cache(maxsize=None)
def _fixture(element, valence, sym="oh"):
    return preload_fixture(element, valence, sym)


@functools.lru_cache(maxsize=None)
def _scratch(element, valence, sym, ct):
    return preload_from_scratch(element, valence, sym, charge_transfer=ct)


def _metal_holes(cache, level):
    """Configuration-weighted 3d hole count of one ground level."""
    holes = {}
    for cfg in cache.decomposition.configs:
        d = [n for l, n in cfg.shells if l == 2]
        holes[configuration_label(cfg.shells)] = 10 - (d[0] if d else 0)
    return sum(w * holes[name] for name, w in level.configurations.items())


def _ratio(cache, **kw):
    _, _, sticks = calcXAS_cached(cache, return_sticks=True, T=T, max_gs=1, **kw)
    analysis = analyze_ground_state(cache, cf=kw.get("cf"), delta=kw.get("delta"), u=kw.get("u"),
                                    lmct=kw.get("lmct"), T=T, n_levels=1, degeneracy_tol=1e-4)
    level = analysis.levels[0]
    return float(sticks[:, 1].sum()) / (0.2 * _metal_holes(cache, level)), level.degeneracy


def _fixture_cases():
    for element, valence in FIXTURES:
        for regime in REGIMES:
            yield pytest.param(element, valence, regime, id=f"{element}{valence}-{regime}")


@pytest.mark.parametrize("element,valence,regime", list(_fixture_cases()))
def test_fixture_intensity_is_an_integer_multiple_of_the_hole_count(element, valence, regime):
    ratio, _ = _ratio(_fixture(element, valence), **REGIMES[regime])
    assert ratio == pytest.approx(round(ratio), abs=1e-5)
    assert round(ratio) >= 1


@pytest.mark.parametrize("element,valence", [
    ("Ti", "iv"), ("V", "iii"), ("Fe", "ii"), ("Co", "ii"), ("Ni", "ii"),
    # the residual-15 cases: 6 ground components split by 2.4e-8 eV in the store
    ("Mn", "ii"), ("Fe", "iii"),
])
def test_fixture_counts_every_ground_component(element, valence):
    ratio, degeneracy = _ratio(_fixture(element, valence), **REGIMES["ionic_10dq0.1"])
    assert round(ratio) == degeneracy


SCRATCH = [
    ("V", "iii", "oh", False, {"cf": {"tendq": 1.8}}),
    # odd electron counts (WP-B2a-3): the legacy half-integer TRANSI emission
    # gave 4.5-10.2 here, which is what flagged it
    ("Cr", "iii", "oh", False, {"cf": {"tendq": 1.8}}),
    ("Mn", "ii", "oh", False, {"cf": {"tendq": 1.0}}),
    ("Fe", "iii", "oh", False, {"cf": {"tendq": 1.2}}),
    ("Co", "ii", "oh", False, {"cf": {"tendq": 1.0}}),
    # d9 has no bundled fixture, so the sum rule is its only oracle; it is also
    # the one ion whose coupling graph needs a dipole component past the first
    # to gauge-fix every node (double_group.gauged_basis)
    ("Cu", "ii", "oh", False, {"cf": {"tendq": 1.0}}),
    # half-integer D4h away from the Oh limit (WP-B2a-4): no Fortran output
    # exists there, so the sum rule and the scramble harness are the oracles
    ("Cr", "iii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Fe", "iii", "d4h", False, {"cf": {"tendq": 1.5, "dt": -0.1, "ds": 0.25}}),
    ("Co", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Cu", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Fe", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Ni", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Ni", "ii", "oh", True, {"cf": {"tendq": 1.1}, "delta": 3.0, "u": 1.0, "lmct": {"eg": 2.0, "t2g": 1.0}}),
    ("Fe", "ii", "d4h", True, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}, "delta": 3.0, "u": 1.0,
                               "lmct": {"b1": 2.0, "a1": 1.6, "b2": 1.0, "e": 0.8}}),
    # charge transfer at odd electron counts (WP-B2a-5). The hole count is now
    # configuration-weighted over two configurations *and* the dipole is the
    # two-channel half-integer one, so this is the invariant that would catch a
    # channel basis rotated between the configurations.
    ("Cr", "iii", "oh", True, {"cf": {"tendq": 1.2}, "delta": 3.0, "u": 1.0, "lmct": {"eg": 2.0, "t2g": 1.0}}),
    ("Fe", "iii", "oh", True, {"cf": {"tendq": 1.1}, "delta": 3.0, "u": 1.0, "lmct": {"eg": 2.0, "t2g": 1.0}}),
    # half-integer D4h + CT has no Fortran reference at all (residual 32)
    ("Fe", "iii", "d4h", True, {"cf": {"tendq": 1.5, "dt": -0.1, "ds": 0.25}, "delta": 3.0, "u": 1.0,
                                "lmct": {"b1": 2.0, "a1": 1.6, "b2": 1.0, "e": 0.8}}),
    # Td (WP-B5-9), integer and half-integer. Td has no Fortran reference and
    # never will have one short of WP-B6, so the sum rule is one of its two
    # oracles; the other is exact equality with Oh (residual 48). A negative
    # tendq is used for one of them because that is what a physical tetrahedron
    # needs -- `tendq` is the coefficient of the cubic invariant in every group,
    # so the sign is the user's to choose, and the sum rule must hold either way.
    ("Ni", "ii", "td", False, {"cf": {"tendq": 1.0}}),
    ("Ni", "ii", "td", False, {"cf": {"tendq": -0.45}}),
    ("Co", "ii", "td", False, {"cf": {"tendq": -0.45}}),
]


SCRATCH_IDS = [f"{s[0]}{s[1]}-{s[2]}{'-ct' if s[3] else ''}" for s in SCRATCH]
# The Oh d5 ions were the residual-15 cases here: 6 ground components split by
# 2.6e-7 (Mn2+) and 7.9e-7 eV (Fe3+), of which exact-float counting kept 2.
# Their D4h runs never were -- the tetragonal field lifts the degeneracy to a
# clean Kramers pair.
SCRATCH_COUNTS = SCRATCH


@pytest.mark.parametrize("element,valence,sym,ct,kw", SCRATCH, ids=SCRATCH_IDS)
def test_from_scratch_intensity_is_an_integer_multiple_of_the_hole_count(element, valence, sym, ct, kw):
    ratio, _ = _ratio(_scratch(element, valence, sym, ct), **kw)
    # absolute energies near 800 eV: eigh noise ~1e-9 eV reaches the 80 K Boltzmann factors at ~1e-7
    assert ratio == pytest.approx(round(ratio), abs=1e-6)
    assert round(ratio) >= 1


@pytest.mark.parametrize("element,valence,sym,ct,kw", SCRATCH_COUNTS, ids=SCRATCH_IDS)
def test_from_scratch_counts_every_ground_component(element, valence, sym, ct, kw):
    ratio, degeneracy = _ratio(_scratch(element, valence, sym, ct), **kw)
    assert round(ratio) == degeneracy
