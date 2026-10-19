"""The one-call convenience API for from-scratch charge transfer, which is missing.

**Not a "not implemented" file any more, and the reason strings used to say so
wrongly.** From-scratch charge transfer *is* implemented and Fortran-validated:
WP-C landed it for integer J (2026-09-15) and WP-B2a-5 for odd electron counts
(2026-09-24), reproducing all four odd-n Oh fixtures at cosine 1.00000000. What
is missing is only the single-call route: ``calcXAS_from_scratch`` takes no
``delta``/``u``/``lmct``/``mlct``, so the supported path is
``preload_from_scratch(..., charge_transfer=True)`` followed by
``calcXAS_cached``. These two xfails therefore record a **usability gap, not an
absent capability** (plan residual 42), and a reader should not conclude from
them that the engine cannot do charge transfer from scratch.

The phased-plan file these used to cite, ``~/.claude/plans/lexical-toasting-
kahan.md``, no longer exists; the current plan is ``docs/DEVELOPMENT_PLAN_2026-09.md``.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="calcXAS_from_scratch exposes no CT kwargs; the engine does "
                           "support CT from scratch via preload_from_scratch (residual 42)")
def test_calcXAS_from_scratch_accepts_ct_kwargs():
    """calcXAS_from_scratch should accept delta/lmct/mlct kwargs and dispatch to CT path."""
    from multitorch.api.calc import calcXAS_from_scratch

    x, y = calcXAS_from_scratch(
        "Ni", "ii",
        cf={"tendq": 1.0, "ds": 0.0, "dt": 0.0}, sym="d4h",
        delta=2.0, lmct=1.5,
    )
    assert x.shape == y.shape


@pytest.mark.xfail(reason="same single-call gap; CT-from-scratch parity itself is covered "
                           "by test_ct_from_scratch_fortran_parity.py (residual 42)")
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
