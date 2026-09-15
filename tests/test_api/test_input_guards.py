"""Input validation added after FABLE_HANDOFF_2026-09-15 (S6 follow-ups, N4, N5, HFS).

Each of these inputs used to run and return a number: a wrong one (T = 0 with
max_gs > 1, half-integer from scratch), a meaningless one (soc = -1, a
string slater, section-0 overrides), or one computed from an unconverged SCF.
"""
from __future__ import annotations

import functools

import pytest
import torch

import multitorch
from multitorch.api.calc import (
    HFS_STALL_TOL, _checked, calcXAS_cached, preload_fixture, preload_from_scratch,
)


@functools.lru_cache(maxsize=None)
def _ni(sym="d4h"):
    return preload_fixture("Ni", "ii", sym)


def _run(cache, **kw):
    return calcXAS_cached(cache, return_sticks=True, nbins=200, **kw)


@pytest.mark.parametrize("T,max_gs", [(0.0, 10), (80.0, 0), (80.0, -1), (-5.0, 1), (80.0, 1.5), (80.0, True)])
def test_boltzmann_pool_is_validated(T, max_gs):
    with pytest.raises(ValueError):
        _run(_ni(), T=T, max_gs=max_gs)


def test_zero_temperature_with_one_level_still_works():
    _, _, sticks = _run(_ni(), T=0.0, max_gs=1)
    assert float(sticks[:, 1].sum()) > 0


@pytest.mark.parametrize("name,value,error", [
    ("soc", -1.0, ValueError), ("slater", float("nan"), ValueError), ("slater", "0.8", TypeError),
    ("soc", True, TypeError), ("slater", torch.tensor([0.8, -0.1]), ValueError),
])
def test_reductions_are_validated(name, value, error):
    with pytest.raises(error):
        _run(_ni(), **{name: value})


def test_tensor_reductions_keep_their_gradient():
    slater = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    _, y = calcXAS_cached(_ni("oh"), slater=slater, nbins=200)
    (g,) = torch.autograd.grad(y.sum(), slater)
    assert torch.isfinite(g) and abs(float(g)) > 0


@pytest.mark.parametrize("sym,cf", [("d4h", {"tdq": 1.0}), ("oh", {"tendq": 1.0, "dt": 0.1}), ("oh", {"ds": 0.1})])
def test_fixture_crystal_field_keys_are_validated(sym, cf):
    with pytest.raises(ValueError, match="crystal-field"):
        _run(_ni(sym), cf=cf)


def test_fixture_overrides_of_unused_sections_raise():
    cache = _ni("oh")
    with pytest.raises(ValueError, match="sections 0/1"):
        _run(cache, atomic={"0.GROUND": {"zeta_1": 0.1}})
    _run(cache, atomic={"2.GROUND": {"zeta_1": 0.1}})   # the ones the assembler reads still work


@pytest.mark.parametrize("element,valence,sym,error", [
    ("Fe", "iii", "oh", NotImplementedError), ("Cu", "ii", "d4h", NotImplementedError),
    ("Ti", "iv", "oh", ValueError), ("Zn", "ii", "oh", ValueError), ("Ni", "ii", "ohh", ValueError),
])
def test_from_scratch_requests_are_rejected_before_hfs(monkeypatch, element, valence, sym, error):
    import multitorch.api.calc as calc

    def no_hfs(*args, **kwargs):
        raise AssertionError("HFS ran before the request was validated")
    monkeypatch.setattr(calc, "_hfs_to_slater_params", no_hfs)
    with pytest.raises(error):
        preload_from_scratch(element, valence, sym)


def test_generator_refuses_half_integer_oh():
    from multitorch.angular.rac_generator import generate_ledge_template
    with pytest.raises(NotImplementedError, match="half-integer"):
        generate_ledge_template(2, 5, sym="oh")


def test_unconverged_hfs_raises_but_a_stalled_limit_cycle_is_accepted():
    from multitorch.atomic.hfs import HFSResult
    ok = HFSResult(orbitals=[], r=torch.zeros(1), IDB=0, Z=28, ION=1, MESH=1, converged=False, n_iter=130, delta=3.7e-6)
    assert _checked(ok, 28, {}) is ok
    bad = HFSResult(orbitals=[], r=torch.zeros(1), IDB=0, Z=28, ION=1, MESH=1, converged=False, n_iter=130,
                    delta=10 * HFS_STALL_TOL)
    with pytest.raises(RuntimeError, match="did not converge"):
        _checked(bad, 28, {})


def test_preload_from_scratch_is_exported():
    assert multitorch.preload_from_scratch is preload_from_scratch
