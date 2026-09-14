"""Ground-state character (multitorch.analysis.ground_state) against analytic limits.

With spin-orbit coupling off, S is a good quantum number and every eigenstate
has ⟨S²⟩ = S(S+1) exactly; the free-ion ground term and its degeneracy follow
from Hund's rules, and the strong-field limits from the t2g/eg filling. The
scalar-operator assembly itself is checked by the identity: √(2J+1)·I on every
J block must come out as the identity in every ground irrep.
"""
from __future__ import annotations

import pytest
import torch

from multitorch.analysis.ground_state import _ground_configs, analyze_ground_state, assemble_scalar_operator
from multitorch.api.calc import _cache_ban, preload_fixture, preload_from_scratch


@pytest.fixture(scope="module")
def caches():
    return {
        "ni_oh": preload_from_scratch("Ni", "ii", "oh"),
        "ni_d4h": preload_from_scratch("Ni", "ii", "d4h"),
        "fe2_oh": preload_from_scratch("Fe", "ii", "oh"),
        "fix_fe2": preload_fixture("Fe", "ii", "oh"),
        "fix_fe3": preload_fixture("Fe", "iii", "oh"),
    }


@pytest.mark.parametrize("key,ct,tol", [("ni_d4h", {}, 1e-12), ("fe2_oh", {}, 1e-12),
                                        ("fix_fe2", dict(delta=3.0, lmct=2.0), 1e-7),
                                        ("fix_fe3", dict(delta=3.0, lmct=2.0), 1e-7)])
def test_identity_assembles_to_identity(caches, key, ct, tol):
    """Fixture tolerance: the Fortran ADD coefficients are printed to finite precision."""
    cache = caches[key]
    ban = _cache_ban(cache, {"tendq": 1.0}, ct.get("delta"), None, ct.get("lmct"), None)
    ones = {i: {sl: 1.0 for J in cache.decomposition.configs[i].states for sl in cache.decomposition.configs[i].states[J]}
            for i in _ground_configs(cache, ban)}
    for sym, X in assemble_scalar_operator(cache, ban, ones).items():
        assert torch.allclose(X, torch.eye(X.shape[0], dtype=X.dtype), atol=tol), sym


# (cache, cf, extra, expected <S^2>, degeneracy, leading term)
LIMITS = [
    ("ni_oh", {"tendq": 0.0}, {}, 2.0, 21, "d8 | 3F"),            # free ion 3F
    ("fe2_oh", {"tendq": 0.0}, {}, 6.0, 25, "d6 | 5D"),           # free ion 5D
    ("fe2_oh", {"tendq": 4.0}, {}, 0.0, 1, None),                # low-spin t2g^6, 1A1g
    ("fe2_oh", {"tendq": 1.0}, {}, 6.0, 15, "d6 | 5D"),           # high-spin 5T2g
    ("fix_fe3", {"tendq": 0.5}, dict(delta=100.0, lmct=0.0), 35 / 4, 6, "d5 | 6S"),  # 6A1g
    ("fix_fe3", {"tendq": 6.0}, dict(delta=100.0, lmct=0.0), 3 / 4, 6, None),       # low-spin 2T2g
    ("fix_fe2", {"tendq": 1.0}, dict(delta=3.0, u=1.0, lmct=2.0), 6.0, 15, None),  # 5T2g with d7L mixing (S conserved)
]


@pytest.mark.parametrize("key,cf,extra,s2,deg,term", LIMITS)
def test_spin_is_exact_without_spin_orbit(caches, key, cf, extra, s2, deg, term):
    tol = 1e-4 if key.startswith("fix") else 1e-9  # fixture RMEs carry 6-decimal print noise
    a = analyze_ground_state(caches[key], cf=cf, slater=0.8, soc=0.0, T=0.0, degeneracy_tol=tol, **extra)
    g = a.levels[0]
    assert g.S2 == pytest.approx(s2, abs=1e-6)
    assert g.degeneracy == deg
    assert sum(g.configurations.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(g.terms.values()) == pytest.approx(1.0, abs=1e-6)
    if term:
        assert g.terms[term] == pytest.approx(1.0, abs=1e-6)
    if extra.get("lmct"):
        weights = list(g.configurations.values())
        assert min(weights) > 0.01  # genuinely mixed; <S^2> stays exact because hopping conserves S


def test_spin_orbit_mixes_spin_but_keeps_weights_normalised(caches):
    a = analyze_ground_state(caches["fe2_oh"], cf={"tendq": 1.0}, slater=0.8, soc=1.0, T=15.0)
    for lv in a.levels:
        assert sum(lv.terms.values()) == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= lv.S2 <= 6.0 + 1e-9
    assert 5.5 < a.levels[0].S2 < 6.0  # mostly 5D with spin-orbit admixture
