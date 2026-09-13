"""WP-A4: autograd == central finite difference for every from-scratch leaf.

Mirror of test_autograd_fd.py (fixture path) for ``calcXAS_from_scratch``.
Leaves: the global reductions ``slater``/``soc``; every atomic integral of both
configurations as an absolute ``atomic`` override (ground 3dⁿ: F²dd, F⁴dd, ζd;
excited 2p⁵3dⁿ⁺¹: F²dd, F⁴dd, F²pd, G¹pd, G³pd, ζp, ζd); 10Dq, and Dt, Ds in
D4h. An override replaces HFS × slater for that integral, so the reductions
and the overrides are tested in separate cases. Loss: fixed-weight sum of the
broadened spectrum on a pinned grid, normalised to O(1). h = 1e-4,
rel ≤ 1e-5 (the S8 contract).

Near-degenerate coverage stays *off* exact ground-level crossings (dt = ds =
1e-3 with a T = 80 K pool; ζ scale 1e-3): with pyctm's unnormalised Boltzmann
weights the spectrum has one-sided derivatives exactly at dt = ds = 0 or ζ = 0
(decision D-1 and Known residual 18 in docs/DEVELOPMENT_PLAN_2026-09.md), so no
finite-difference oracle exists there.

Also at the contraction seam: ∂H/∂p_i is the operator block O_i, ∂H/∂slater
is Σ_Slater HFS_i·O_i and ∂H/∂soc is Σ_ζ HFS_i·O_i (gradient isolation).
"""
from __future__ import annotations

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.api.calc import _from_scratch_structure, calcXAS_from_scratch
from multitorch.hamiltonian.parametric import is_soc, rebuild_hamiltonian_store

H = 1e-4
REL_TOL = 1e-5

GRIDS = {("Ni", "ii"): dict(xmin=-15.0, xmax=25.0, nbins=1600),
         ("Fe", "ii"): dict(xmin=-15.0, xmax=30.0, nbins=1800)}

ATOMIC = {"gs": ("F2dd", "F4dd", "zeta_d"),
          "ex": ("F2dd", "F4dd", "F2pd", "G1pd", "G3pd", "zeta_p", "zeta_d")}

# Analytically inert leaves: Ni²⁺'s excited 3d⁹ is the single term ²D, so the
# d–d Coulomb integrals only shift the configuration average (removed in
# Cowan's E_av convention) and their operators vanish identically.
INERT = {("Ni", "ii"): {"ex.F2dd", "ex.F4dd"}}


def _atomic_at(element, valence, sym, slater=0.8, soc=1.0):
    """HFS × (slater | soc) for every override name, as floats."""
    _, _, dec = _from_scratch_structure(element, valence, sym, "blume_watson")
    out = {}
    for label, names in ATOMIC.items():
        cfg = dec.by_label(label)
        values, alias = cfg.parameter_values(slater, soc), cfg.aliases()
        for n in names:
            out[f"{label}.{n}"] = values[alias[n]]
    return out


def _spectrum(element, valence, sym, p, sticks=None):
    cf = {k: p[k] for k in ("tendq", "dt", "ds") if k in p}
    atomic = {}
    for key, v in p.items():
        if "." in key:
            label, name = key.split(".")
            atomic.setdefault(label, {})[name] = v
    _, y = calcXAS_from_scratch(element, valence, cf, slater=p.get("slater", 0.8), soc=p.get("soc", 1.0),
                                sym=sym, atomic=atomic or None, **(sticks or {}), **GRIDS[(element, valence)])
    return y


CF = {"oh": dict(tendq=1.0), "d4h": dict(tendq=1.0, dt=0.05, ds=0.1)}
SYSTEMS = [("Ni", "ii", "oh"), ("Ni", "ii", "d4h"), ("Fe", "ii", "oh")]


POOL = dict(T=80.0, max_gs=40)


def _cases():
    for el, val, sym in SYSTEMS:
        yield pytest.param(el, val, sym, "scale", {}, None, id=f"{el}{val}_{sym}_scale")
        yield pytest.param(el, val, sym, "atomic", {}, None, id=f"{el}{val}_{sym}_atomic")
    # near (not at) ground-level crossings, decision D-1
    yield pytest.param("Ni", "ii", "d4h", "scale", dict(dt=1e-3, ds=1e-3), POOL, id="Niii_d4h_dtds1e-3_pool")
    yield pytest.param("Fe", "ii", "oh", "scale", dict(soc=1e-3), POOL, id="Feii_oh_soc1e-3_pool")


@pytest.mark.parametrize("element,valence,sym,mode,shift,sticks", list(_cases()))
def test_from_scratch_autograd_matches_finite_difference(element, valence, sym, mode, shift, sticks):
    base = dict(CF[sym])
    if mode == "scale":
        base.update(slater=0.8, soc=1.0)
    else:
        base.update(_atomic_at(element, valence, sym))
    base.update(shift)
    nbins = GRIDS[(element, valence)]["nbins"]
    w = torch.linspace(0.5, 1.5, nbins, dtype=DTYPE)
    loss_of = lambda p: (w * _spectrum(element, valence, sym, p, sticks)).sum()
    scale = float(loss_of(base))

    leaves = {k: torch.tensor(float(v), dtype=DTYPE, requires_grad=True) for k, v in base.items()}
    grads = torch.autograd.grad(loss_of(leaves) / scale, list(leaves.values()))

    inert = INERT.get((element, valence), set())
    for (name, value), g in zip(base.items(), grads):
        plus, minus = dict(base), dict(base)
        plus[name], minus[name] = value + H, value - H
        fd = float((loss_of(plus) - loss_of(minus)) / (2 * H) / scale)
        assert torch.isfinite(g), name
        if name in inert:
            assert fd == 0.0 and float(g) == 0.0, (name, float(g), fd)
            continue
        assert abs(fd) > 1e-6, (name, fd)  # every other leaf moves this spectrum
        assert abs(float(g) - fd) <= REL_TOL * abs(fd), (name, float(g), fd)


@pytest.mark.parametrize("sym", ["oh", "d4h"])
def test_seam_gradients_are_operator_blocks(sym):
    _, template, dec = _from_scratch_structure("Ni", "ii", sym, "blume_watson")
    gen = torch.Generator().manual_seed(0)
    slater = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
    soc = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    f2pd = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)
    store = rebuild_hamiltonian_store(template, dec, slater=slater, soc=soc, atomic={"ex": {"F2pd": f2pd}})

    for cfg in dec.configs:
        W = {J: torch.randn(*store[0][j].shape, generator=gen, dtype=DTYPE) for J, j in cfg.block_index.items()}
        loss = sum((W[J] * store[0][j]).sum() for J, j in cfg.block_index.items())
        g_sl, g_soc, g_f2 = torch.autograd.grad(loss, [slater, soc, f2pd], allow_unused=True)
        contract = lambda name: sum(float((W[J] * cfg.operators[name][J]).sum()) for J in cfg.block_index)
        overridden = {"F2_12"} if cfg.label == "ex" else set()
        want_sl = sum(cfg.reference[n] * contract(n) for n in cfg.operators if not is_soc(n) and n not in overridden)
        want_soc = sum(cfg.reference[n] * contract(n) for n in cfg.operators if is_soc(n))
        assert float(g_sl) == pytest.approx(want_sl, rel=1e-12, abs=1e-12)
        assert float(g_soc) == pytest.approx(want_soc, rel=1e-12, abs=1e-12)
        if cfg.label == "ex":
            assert float(g_f2) == pytest.approx(contract("F2_12"), rel=1e-12)
        else:
            assert g_f2 is None
