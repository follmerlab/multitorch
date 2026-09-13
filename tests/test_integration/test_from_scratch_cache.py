"""WP-A6: from-scratch caches drive calcXAS_cached / calcXAS_batch.

The Fortran oracle for the cached from-scratch path with explicit atomic
parameters is in test_from_scratch_fortran_parity.py; the gradient contract in
test_autograd_fd_from_scratch.py (calcXAS_from_scratch is preload + cached).
Here: batch == per-sample, the BAN crystal-field convention, and refusals.
"""
from __future__ import annotations

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcXAS_batch, calcXAS_cached, preload_from_scratch

GRID = dict(xmin=-15.0, xmax=25.0, nbins=800)


@pytest.fixture(scope="module")
def ni_d4h():
    return preload_from_scratch("Ni", "ii", "d4h")


def test_batch_equals_per_sample(ni_d4h):
    cf = {"tendq": 1.1, "dt": 0.04, "ds": -0.08}
    s = torch.tensor([0.6, 0.8, 1.0], dtype=DTYPE)
    z = torch.tensor([0.7, 1.0, 1.2], dtype=DTYPE)
    atomic = {"ex": {"G1pd": 4.0}}
    yb = calcXAS_batch(ni_d4h, s, z, cf=cf, atomic=atomic, **GRID)
    for i in range(3):
        _, y = calcXAS_cached(ni_d4h, cf=cf, slater=float(s[i]), soc=float(z[i]), atomic=atomic, **GRID)
        assert torch.allclose(yb[i], y, rtol=0, atol=1e-12 * float(y.abs().max()))


def test_crystal_field_is_ballhausen_not_the_fixture_slot(ni_d4h):
    """From scratch, XHAM holds 10Dq itself; the fixture BAN holds 10Dq − 35·Dt/6.

    Oracle: one-electron probe (test_cf_one_electron.py) pins the from-scratch
    operators to Ballhausen; here the cache must hand them the raw values.
    """
    from multitorch.api.calc import _cache_ban

    ban = _cache_ban(ni_d4h, {"tendq": 1.0, "dt": 0.1, "ds": 0.2}, None, None, None, None)
    assert ban.xham[0].values == [1.0, 1.0, 0.1, 0.2]
    assert ni_d4h.ban.xham[0].values[1:] == [1.0, 0.0, 0.0]  # template untouched


def test_charge_transfer_and_unknown_keys_raise(ni_d4h):
    for kw in (dict(delta=3.0), dict(u=1.0), dict(lmct=2.0)):
        with pytest.raises(ValueError):
            calcXAS_cached(ni_d4h, cf={"tendq": 1.0}, **kw, **GRID)
    with pytest.raises(ValueError):
        calcXAS_cached(ni_d4h, cf={"10dq": 1.0}, **GRID)


def test_batch_gradients_are_per_sample(ni_d4h):
    s = torch.tensor([0.7, 0.9], dtype=DTYPE, requires_grad=True)
    z = torch.tensor([1.0, 1.0], dtype=DTYPE)
    w = torch.linspace(0.5, 1.5, GRID["nbins"], dtype=DTYPE)
    yb = calcXAS_batch(ni_d4h, s, z, cf={"tendq": 1.0}, **GRID)
    (g,) = torch.autograd.grad((w * yb[1]).sum(), [s])
    assert g[0] == 0.0 and g[1] != 0.0
