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
10Dq 0.1: the second PRMULT copy of S1+→S1− was never assembled; fixed). Still
open:

* residual 15 (exact-float level counting at ``max_gs=1``): an integer, but
  below the ground degeneracy (Mn²⁺, Fe³⁺ count 2 of 6 near-degenerate
  components split by µeV).
"""
from __future__ import annotations

import functools

import pytest

from multitorch.analysis import analyze_ground_state
from multitorch.analysis.ground_state import configuration_label
from multitorch.api.calc import calcXAS_cached, preload_fixture, preload_from_scratch

T = 80.0
RES15 = pytest.mark.xfail(strict=True, reason="plan residual 15: max_gs counts exactly equal levels only")

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
    pytest.param("Mn", "ii", marks=RES15), pytest.param("Fe", "iii", marks=RES15),
])
def test_fixture_counts_every_ground_component(element, valence):
    ratio, degeneracy = _ratio(_fixture(element, valence), **REGIMES["ionic_10dq0.1"])
    assert round(ratio) == degeneracy


SCRATCH = [
    ("V", "iii", "oh", False, {"cf": {"tendq": 1.8}}),
    ("Fe", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Ni", "ii", "d4h", False, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}}),
    ("Ni", "ii", "oh", True, {"cf": {"tendq": 1.1}, "delta": 3.0, "u": 1.0, "lmct": {"eg": 2.0, "t2g": 1.0}}),
    ("Fe", "ii", "d4h", True, {"cf": {"tendq": 1.2, "dt": 0.05, "ds": 0.1}, "delta": 3.0, "u": 1.0,
                               "lmct": {"b1": 2.0, "a1": 1.6, "b2": 1.0, "e": 0.8}}),
]


@pytest.mark.parametrize("element,valence,sym,ct,kw", SCRATCH,
                         ids=[f"{s[0]}{s[1]}-{s[2]}{'-ct' if s[3] else ''}" for s in SCRATCH])
def test_from_scratch_intensity_counts_holes_and_components(element, valence, sym, ct, kw):
    ratio, degeneracy = _ratio(_scratch(element, valence, sym, ct), **kw)
    # absolute energies near 800 eV: eigh noise ~1e-9 eV reaches the 80 K Boltzmann factors at ~1e-7
    assert ratio == pytest.approx(round(ratio), abs=1e-6)
    assert round(ratio) == degeneracy
