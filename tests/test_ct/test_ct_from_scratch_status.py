"""Status sentinel for the charge-transfer from-scratch port (plan Phase 2).

Each test is xfail until the corresponding piece of the CT port lands.
See `~/.claude/plans/lexical-toasting-kahan.md` for the phased plan.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="Phase 2: CT from-scratch generator not yet implemented")
def test_calcXAS_from_scratch_accepts_ct_kwargs():
    """calcXAS_from_scratch should accept delta/lmct/mlct kwargs and dispatch to CT path."""
    from multitorch.api.calc import calcXAS_from_scratch

    x, y = calcXAS_from_scratch(
        "Ni", "ii",
        cf={"tendq": 1.0, "ds": 0.0, "dt": 0.0}, sym="d4h",
        delta=2.0, lmct=1.5,
    )
    assert x.shape == y.shape


@pytest.mark.xfail(reason="Phase 2: CT parity test against nid8ct pending")
def test_ct_from_scratch_matches_nid8ct_fixture():
    """CT from-scratch should match the bundled nid8ct fixture."""
    from multitorch.api.calc import calcXAS_cached, calcXAS_from_scratch, preload_fixture
    cache = preload_fixture("Ni", "ii", "d4h")
    _, y_ref = calcXAS_cached(cache, cf={"tendq": 1.0, "ds": 0.0, "dt": 0.0},
                               delta=2.0, lmct=1.5)
    _, y_new = calcXAS_from_scratch(
        "Ni", "ii", cf={"tendq": 1.0, "ds": 0.0, "dt": 0.0}, sym="d4h",
        delta=2.0, lmct=1.5,
    )
    assert y_new.shape == y_ref.shape
