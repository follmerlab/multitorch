"""WP-S S8: autograd == central finite difference for every fixture-path leaf.

Loss: a fixed-weight sum of the broadened spectrum on a pinned grid (auto
ranging would move the grid with the parameters), normalised to O(1). Leaves:
slater, soc, 10Dq, Dt, Ds (D4h), Δ, u and one hopping channel V. Points include
near-degenerate regimes (Dt = Ds = 1e-3, soc = 1e-3) and a crystal-field tensor
exactly at 0, whose gradient the former ``xv != 0.0`` guard severed.
Tolerance 1e-5 relative at h = 1e-4 (observed ≤ 1.2e-6 on 2026-09-13).
"""
from __future__ import annotations

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import calcXAS_cached, preload_fixture

H = 1e-4
REL_TOL = 1e-5


@pytest.fixture(scope="module")
def caches():
    return {"ni_d4h": preload_fixture("Ni", "ii", "d4h"), "fe_oh": preload_fixture("Fe", "ii", "oh")}


GRIDS = {"ni_d4h": dict(xmin=850.0, xmax=885.0, nbins=1400),
         "fe_oh": dict(xmin=5.0, xmax=45.0, nbins=1600)}

NI = dict(slater=0.8, soc=1.0, tendq=1.0, dt=0.05, ds=0.1, delta=5.0, u=1.0, V0=2.0)
FE = dict(slater=0.8, soc=1.0, tendq=1.0, delta=5.0, u=6.5, V0=2.0)

CASES = [
    ("ni_d4h", NI),
    ("ni_d4h", {**NI, "dt": 1e-3, "ds": 1e-3}),
    ("ni_d4h", {**NI, "soc": 1e-3}),
    ("ni_d4h", {**NI, "ds": 0.0}),
    ("fe_oh", FE),
    ("fe_oh", {**FE, "soc": 1e-3}),
]


def _spectrum(cache, p, grid):
    names = ("eg", "t2g") if cache.sym == "oh" else ("b1", "a1", "b2", "e")
    cf = {k: p[k] for k in ("tendq", "dt", "ds") if k in p}
    _, y = calcXAS_cached(cache, cf=cf, slater=p["slater"], soc=p["soc"],
                          delta=p["delta"], u=p["u"], lmct={names[0]: p["V0"]}, **grid)
    return y


@pytest.mark.parametrize("key,base", CASES,
                         ids=["ni_d4h", "ni_d4h_dtds1e-3", "ni_d4h_soc1e-3", "ni_d4h_ds0",
                              "fe_oh", "fe_oh_soc1e-3"])
def test_autograd_matches_finite_difference(caches, key, base):
    cache, grid = caches[key], GRIDS[key]
    w = torch.linspace(0.5, 1.5, grid["nbins"], dtype=DTYPE)
    scale = float((w * _spectrum(cache, base, grid)).sum())

    leaves = {k: torch.tensor(float(v), dtype=DTYPE, requires_grad=True) for k, v in base.items()}
    loss = (w * _spectrum(cache, leaves, grid)).sum() / scale
    grads = torch.autograd.grad(loss, list(leaves.values()))

    for (name, value), g in zip(base.items(), grads):
        plus, minus = dict(base), dict(base)
        plus[name], minus[name] = value + H, value - H
        fd = float(((w * _spectrum(cache, plus, grid)).sum()
                    - (w * _spectrum(cache, minus, grid)).sum()) / (2 * H) / scale)
        assert torch.isfinite(g), name
        assert abs(float(g) - fd) <= REL_TOL * max(abs(fd), 1e-8), (name, float(g), fd)
