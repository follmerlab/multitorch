"""The one-call route to from-scratch charge transfer (plan residual 42).

**Not a "not implemented" file, and the reason strings used to say so wrongly.**
From-scratch charge transfer *is* implemented and Fortran-validated: WP-C
landed it for integer J (2026-09-15) and WP-B2a-5 for odd electron counts
(2026-09-24), reproducing all four odd-n Oh fixtures at cosine 1.00000000. What
was missing until 2026-10-19 was only the convenience: ``calcXAS_from_scratch``
took no ``delta``/``u``/``lmct``/``mlct``, so the supported route was
``preload_from_scratch(..., charge_transfer=True)`` followed by
``calcXAS_cached``. It now takes all four and preloads with
``charge_transfer=True`` whenever any of them is given.

**What these tests can and cannot be.** The physics oracle for from-scratch CT
is the Fortran one in ``test_ct_from_scratch_fortran_parity.py`` — four odd-n Oh
fixtures at cosine 1.00000000 — and it is not repeated here. What is left to
assert is the thing the new keywords themselves promise: that the one-call route
is *exactly* the two-step route, bit for bit, and that its argument handling is
not silently dropping a parameter. That is an API-equivalence claim, not a
physics claim, which is why ``torch.equal`` against the two-step route is the
right test here and a weaker "the shapes agree" is not. The file this replaced
asserted only ``y_new.shape == y_ref.shape`` on a default 2000-bin grid, which
is true however wrong the spectrum is.

The phased-plan file these used to cite, ``~/.claude/plans/lexical-toasting-
kahan.md``, no longer exists; the current plan is ``docs/DEVELOPMENT_PLAN_2026-09.md``.
"""
from __future__ import annotations

import functools

import pytest
import torch

CF = {"tendq": 1.0, "ds": 0.0, "dt": 0.0}
LMCT = {"b1": 1.5, "a1": 1.2, "b2": 1.0, "e": 0.8}


@functools.lru_cache(maxsize=None)
def _two_step_cache():
    from multitorch.api.calc import preload_from_scratch
    return preload_from_scratch("Ni", "ii", "d4h", charge_transfer=True)


def test_calcXAS_from_scratch_accepts_ct_kwargs():
    """The four CT keywords are accepted and produce a finite spectrum."""
    from multitorch.api.calc import calcXAS_from_scratch

    x, y = calcXAS_from_scratch(
        "Ni", "ii", cf=CF, sym="d4h", delta=2.0, lmct=1.5,
    )
    assert x.shape == y.shape
    assert torch.isfinite(y).all()
    assert float(y.sum()) > 0.0


def test_the_one_call_route_is_exactly_the_two_step_route():
    """``calcXAS_from_scratch`` with CT keywords == preload + ``calcXAS_cached``.

    Bit-identical, not merely close: the one-call form is meant to *be* the
    two-step form, so any parameter it fails to forward, or forwards in a
    different order, shows up here. Every keyword of the four is given a
    distinct value, and ``lmct`` is a per-channel dict so a channel dropped on
    the way through is caught too.
    """
    from multitorch.api.calc import calcXAS_cached, calcXAS_from_scratch

    kw = dict(delta=2.0, u=1.3, lmct=LMCT, T=120.0, max_gs=4, nbins=600)
    x_ref, y_ref = calcXAS_cached(_two_step_cache(), cf=CF, **kw)
    x_new, y_new = calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", **kw)

    assert torch.equal(x_new, x_ref)
    assert torch.equal(y_new, y_ref)


def test_ct_keywords_switch_the_preload_and_change_the_spectrum():
    """CT is really added, not accepted and ignored.

    Without the keywords the cache is single-configuration; with them it has
    the ligand-hole configurations as well, so the spectrum must move. A
    no-op implementation that accepted the keywords and preloaded without
    charge transfer would pass the two tests above (the two-step route would
    be no-op too) and fails here.
    """
    from multitorch.api.calc import calcXAS_from_scratch

    grid = dict(xmin=-15.0, xmax=30.0, nbins=600)
    _, y_plain = calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", **grid)
    _, y_ct = calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h",
                                   delta=2.0, u=1.3, lmct=LMCT, **grid)
    assert not torch.allclose(y_plain, y_ct, atol=1e-8)


def test_delta_carries_a_gradient_through_the_one_call_route():
    """The docstring claims differentiability in the CT parameters; check it.

    Autograd-vs-finite-difference for every CT leaf is
    ``test_autograd_fd_ct_from_scratch.py``'s job; this only asserts the
    one-call route does not break the graph on its way in.
    """
    from multitorch._constants import DTYPE
    from multitorch.api.calc import calcXAS_from_scratch

    delta = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    _, y = calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", delta=delta,
                                u=1.3, lmct=LMCT, xmin=-15.0, xmax=30.0, nbins=600)
    (g,) = torch.autograd.grad(y.sum(), delta)
    assert torch.isfinite(g) and abs(float(g)) > 1e-8


def test_ct_keywords_without_delta_are_refused_before_the_preload(monkeypatch):
    """``u``/``lmct`` alone must raise, and raise *before* the expensive part.

    ``_cache_ban`` would refuse it anyway, but only after the HFS and angular
    preload, so the user would wait a few seconds to be told that a required
    argument is missing. The check therefore has to come first, and "first" is
    asserted rather than asserted-about: ``preload_from_scratch`` is replaced
    by something that fails loudly, so a check that moved below the preload
    would raise ``AssertionError`` here instead of ``ValueError``.
    """
    from multitorch.api import calc

    def _must_not_run(*a, **k):
        raise AssertionError("preloaded before validating delta")

    monkeypatch.setattr(calc, "preload_from_scratch", _must_not_run)
    for bad in (dict(u=1.0), dict(lmct=1.5), dict(u=1.0, lmct=1.5)):
        with pytest.raises(ValueError, match="need delta"):
            calc.calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", **bad)


def test_mlct_is_refused_rather_than_ignored():
    """There is no MLCT configuration yet (plan: carried forward)."""
    from multitorch.api.calc import calcXAS_from_scratch

    with pytest.raises(ValueError, match="mlct"):
        calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", delta=2.0, mlct=0.5)


def test_no_ct_keywords_still_means_a_single_configuration_calculation():
    """The plain route is unchanged: no CT keywords, no ligand-hole configs."""
    from multitorch.api.calc import preload_from_scratch

    assert preload_from_scratch("Ni", "ii", "d4h").ban.nconf_gs == 1
    assert _two_step_cache().ban.nconf_gs == 2


def test_level_window_reaches_calcXAS_cached(monkeypatch):
    """``level_window`` was the one ``calcXAS_cached`` keyword with no route here.

    Not part of residual 42 itself, but the same gap and the same signature:
    ``calcXAS_cached`` has taken ``level_window`` since residual 37 (the d5
    Boltzmann-pool finding), ``calcXAS`` and ``getXAS`` take it, and
    ``calcXAS_from_scratch`` was the entry point that could not reach it. What
    the window *does* is tested where it was introduced
    (``test_tm_series.py``, against the pyctm oracle at ``level_window=0.0``);
    all that is open here is whether the keyword arrives, so this captures the
    call instead of paying for a d5 spectrum to observe it indirectly.
    """
    from multitorch.api import calc

    seen = {}

    def _capture(cache, **kwargs):
        seen.update(kwargs)
        import torch as _t
        return _t.zeros(2), _t.zeros(2)

    monkeypatch.setattr(calc, "calcXAS_cached", _capture)
    calc.calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h", level_window=0.0)
    assert seen["level_window"] == 0.0

    seen.clear()
    calc.calcXAS_from_scratch("Ni", "ii", cf=CF, sym="d4h")
    assert seen["level_window"] is None       # default, not silently 0.0
