"""WP-C C5: autograd == central finite difference for the charge-transfer leaves from scratch.

Mirror of test_autograd_fd_from_scratch.py for ``preload_from_scratch(...,
charge_transfer=True)`` + ``calcXAS_cached``. Leaves: Δ, u, V of every
hybridisation channel, 10Dq (Dt, Ds in D4h), the reductions ``slater``/``soc``,
and absolute ``atomic`` overrides of the ligand-hole configurations. Loss:
fixed-weight sum of the broadened spectrum on a pinned grid, normalised to
O(1). h = 1e-4, rel ≤ 1e-5 (the S8 contract). A T = 80 K pool keeps the
sticks away from exact ground-level crossings (decision D-1).
"""
from __future__ import annotations

import functools

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcXAS_cached, preload_from_scratch

H = 1e-4
REL_TOL = 1e-5
POOL = dict(T=80.0, max_gs=40)
GRID = dict(xmin=-15.0, xmax=30.0, nbins=1800)

SYSTEMS = {
    "Ni_d4h": ("Ni", "ii", "d4h", dict(tendq=1.2, dt=0.05, ds=0.1, delta=3.0, u=1.0,
                                       b1=2.0, a1=1.6, b2=1.0, e=0.8, slater=0.8, soc=1.0,
                                       **{"gs_lh.zeta_1": 0.07, "ex_lh.zeta_1": 11.0})),
    "Fe_oh": ("Fe", "ii", "oh", dict(tendq=1.0, delta=3.0, u=1.0, eg=2.2, t2g=1.1, slater=0.8, soc=1.0,
                                     **{"gs_lh.F2_11": 7.5, "ex_lh.G1_12": 3.6})),
}
CHANNELS = ("eg", "t2g", "b1", "a1", "b2", "e")


@functools.lru_cache(maxsize=None)
def _cache(element, valence, sym):
    return preload_from_scratch(element, valence, sym, charge_transfer=True)


def _spectrum(element, valence, sym, p):
    atomic = {}
    for key, v in p.items():
        if "." in key:
            label, name = key.split(".")
            atomic.setdefault(label, {})[name] = v
    _, y = calcXAS_cached(
        _cache(element, valence, sym),
        cf={k: p[k] for k in ("tendq", "dt", "ds") if k in p},
        slater=p["slater"], soc=p["soc"], delta=p["delta"], u=p["u"],
        lmct={k: p[k] for k in CHANNELS if k in p}, atomic=atomic, **POOL, **GRID)
    return y


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_charge_transfer_autograd_matches_finite_difference(system):
    element, valence, sym, base = SYSTEMS[system]
    w = torch.linspace(0.5, 1.5, GRID["nbins"], dtype=DTYPE)
    loss_of = lambda p: (w * _spectrum(element, valence, sym, p)).sum()
    scale = float(loss_of(base))

    leaves = {k: torch.tensor(float(v), dtype=DTYPE, requires_grad=True) for k, v in base.items()}
    grads = torch.autograd.grad(loss_of(leaves) / scale, list(leaves.values()))
    for (name, value), g in zip(base.items(), grads):
        plus, minus = dict(base), dict(base)
        plus[name], minus[name] = value + H, value - H
        fd = float((loss_of(plus) - loss_of(minus)) / (2 * H) / scale)
        assert torch.isfinite(g), name
        assert abs(fd) > 1e-6, (name, fd)   # every leaf moves the spectrum
        assert abs(float(g) - fd) <= REL_TOL * abs(fd), (name, float(g), fd)
