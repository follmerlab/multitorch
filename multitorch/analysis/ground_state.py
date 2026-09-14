"""
Ground-state character of a multiplet calculation: spin, LS terms and configurations.

Given a cache (``preload_fixture`` or ``preload_from_scratch``) and the physical
parameters of a calculation, :func:`analyze_ground_state` diagonalises the
ground-state Hamiltonian of every ground irrep and reports, for each of the
lowest levels,

* ``⟨S²⟩`` and the effective spin S with S(S+1) = ⟨S²⟩,
* the weight of every configuration (3dⁿ, 3dⁿ⁺¹L̲, ...),
* the weight of every total (S, L) term within each configuration,

plus Boltzmann averages at a temperature.

Method. Every row of a HAMILTONIAN J block in the COWAN store is a state of
definite total S and L (``ConfigDecomposition.states``). A diagonal operator in
that basis, stored with the same √(2J+1) convention as the Hamiltonian, is a
rotational scalar, so the HAMILTONIAN ADD coefficients that couple the J blocks
into point-group irrep blocks carry it into exactly the basis the ground
eigenvectors live in. S² is diag(S(S+1)); a term projector is diag(1) on the
rows of that term. Both are assembled with the crystal field, energy offsets
and hybridisation switched off, and evaluated as uᵀ X u on each eigenvector.
Spin-orbit coupling mixes S, so ⟨S²⟩ is not an integer spin in general.

Boltzmann averages here are normalised by the partition function (they are
thermal expectation values), unlike the unnormalised spectral weights of
decision D-1.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from multitorch._constants import DTYPE, K_B_FLOAT

_L_LETTER = "SPDFGHIKLMNOQRTUV"


def term_label(S: float, L: float) -> str:
    """``(2, 2)`` → ``'5D'``."""
    return f"{int(round(2 * S + 1))}{_L_LETTER[int(round(L))]}"


def configuration_label(shells) -> str:
    """Open shells ``((2, 7), (2, 9))`` → ``'d7 d9'``."""
    return " ".join(f"{'spdfgh'[l]}{n}" for l, n in shells) or "closed"


@dataclass
class GroundLevel:
    energy: float                      # eV, on the calculation's energy scale
    relative_meV: float                # above the lowest level
    degeneracy: int                    # number of states (irrep dimension x copies)
    irreps: List[str]
    S2: float                          # <S^2>
    S_eff: float                       # S with S(S+1) = <S^2>
    configurations: Dict[str, float]   # configuration label -> weight
    terms: Dict[str, float]            # "config | term" -> weight, largest first


@dataclass
class GroundStateAnalysis:
    levels: List[GroundLevel]
    temperature: float
    thermal_S2: float
    thermal_S_eff: float
    thermal_configurations: Dict[str, float]
    thermal_terms: Dict[str, float] = field(default_factory=dict)

    def summary(self, n: int = 6, min_weight: float = 0.02) -> str:
        lines = [f"T = {self.temperature:g} K: <S^2> = {self.thermal_S2:.3f} (S_eff = {self.thermal_S_eff:.3f}); "
                 + ", ".join(f"{k} {v:.3f}" for k, v in self.thermal_configurations.items())]
        for lv in self.levels[:n]:
            terms = ", ".join(f"{k} {v:.2f}" for k, v in lv.terms.items() if v >= min_weight)
            lines.append(f"  {lv.relative_meV:9.3f} meV  x{lv.degeneracy:<3d} <S^2>={lv.S2:6.3f} "
                         f"S_eff={lv.S_eff:5.3f}  {terms}")
        return "\n".join(lines)


def _s_eff(s2: float) -> float:
    return 0.5 * (math.sqrt(1.0 + 4.0 * max(s2, 0.0)) - 1.0)


def _operator_store(cache, diagonal: Mapping[int, Mapping[Tuple[float, float], float]]):
    """Template store whose decomposed HAMILTONIAN blocks hold √(2J+1)·diag(value(S, L)).

    ``diagonal[i]`` maps (S, L) to the value on configuration ``i`` of the
    decomposition (index into ``cache.decomposition.configs``); configurations
    not listed get a zero block.
    """
    store = [list(sec) for sec in cache.cowan_template]
    for i, cfg in enumerate(cache.decomposition.configs):
        values = diagonal.get(i, {})
        for J, j in cfg.block_index.items():
            d = [values.get(sl, 0.0) for sl in cfg.states[J]]
            store[cfg.section][j] = math.sqrt(2 * J + 1) * torch.diag(torch.tensor(d, dtype=DTYPE))
    return store


def _operator_ban(ban):
    """BAN with only the HAMILTONIAN operator, no offsets and no hybridisation."""
    from multitorch.io.read_ban import XHAMEntry, XMIXEntry

    out = copy.copy(ban)
    out.xham = [XHAMEntry(values=[1.0] + [0.0] * (len(x.values) - 1), combos=list(x.combos)) for x in ban.xham]
    out.xmix = [XMIXEntry(values=[0.0] * len(x.values), combos=list(x.combos)) for x in ban.xmix]
    out.eg = {k: 0.0 for k in ban.eg}
    out.ef = {k: 0.0 for k in ban.ef}
    return out


def assemble_scalar_operator(cache, ban, diagonal) -> Dict[str, torch.Tensor]:
    """Irrep-basis matrices of a scalar operator diagonal in (S, L); see :func:`_operator_store`."""
    from multitorch.hamiltonian.assemble import assemble_ground_hamiltonians

    blocks = assemble_ground_hamiltonians(_operator_store(cache, diagonal), cache.rac, _operator_ban(ban))
    return {sym: H for sym, (H, _, _) in blocks.items()}


def _ground_configs(cache, ban) -> List[int]:
    """Decomposition indices of the ground-manifold configurations, in assembler order (config 1, 2, ...)."""
    section = 2 if ban.nconf_gs >= 2 else 0
    order = {"GROUND": 0, "EXCITE": 1}
    idx = [i for i, c in enumerate(cache.decomposition.configs) if c.section == section]
    if ban.nconf_gs < 2:  # single configuration: the from-scratch 'gs' (GROUND) block only
        idx = [i for i in idx if cache.decomposition.configs[i].block_type == "GROUND"]
    return sorted(idx, key=lambda i: order[cache.decomposition.configs[i].block_type])


def analyze_ground_state(
    cache,
    cf: Optional[dict] = None,
    slater=0.8,
    soc=1.0,
    delta=None, u=None, lmct=None,
    atomic: Optional[dict] = None,
    T: float = 15.0,
    n_levels: int = 12,
    degeneracy_tol: float = 1e-5,
) -> GroundStateAnalysis:
    """Spin, term and configuration character of the lowest ground levels.

    Parameters mirror :func:`~multitorch.api.calc.calcXAS_cached`. Levels closer
    than ``degeneracy_tol`` eV (across irreps) are merged; their weights are
    degeneracy-weighted means. Fixture stores carry the 6-decimal print
    precision of the Fortran RMEs, which splits exact degeneracies by up to
    ~1e-6·|H| (1e-5 to 1e-4 eV); from-scratch stores are exact to ~1e-12 eV.
    Real splittings smaller than the tolerance (e.g. a cubic zero-field
    splitting) are merged too; lower it for those.
    """
    from multitorch.api.calc import _cache_ban
    from multitorch.hamiltonian.assemble import assemble_ground_hamiltonians
    from multitorch.hamiltonian.parametric import rebuild_hamiltonian_store

    with torch.no_grad():
        ban = _cache_ban(cache, cf, delta, u, lmct, None)
        store = rebuild_hamiltonian_store(cache.cowan_template, cache.decomposition,
                                          slater=slater, soc=soc, atomic=atomic)
        hams = assemble_ground_hamiltonians(store, cache.rac, ban)
        dims = {ir.name: ir.dim for ir in cache.rac.irreps}

        cfg_idx = _ground_configs(cache, ban)
        cfgs = [cache.decomposition.configs[i] for i in cfg_idx]
        cfg_names = [configuration_label(c.shells) for c in cfgs]
        s2 = assemble_scalar_operator(
            cache, ban, {i: {sl: sl[0] * (sl[0] + 1) for J in c.states for sl in c.states[J]}
                         for i, c in zip(cfg_idx, cfgs)})
        projectors = {}
        for i, c, name in zip(cfg_idx, cfgs, cfg_names):
            for sl in sorted({sl for J in c.states for sl in c.states[J]}, key=lambda x: (-x[0], x[1])):
                projectors[f"{name} | {term_label(*sl)}"] = assemble_scalar_operator(cache, ban, {i: {sl: 1.0}})

        states = []  # (E, irrep, dim, S2, {config: w}, {term: w})
        for sym, (H, labels, sizes) in hams.items():
            E, U = torch.linalg.eigh(0.5 * (H + H.T))
            S2u = torch.einsum("in,ij,jn->n", U, s2[sym], U)
            cw = [(U[labels == k + 1, :] ** 2).sum(dim=0) for k in range(len(sizes))]
            tw = {t: torch.einsum("in,ij,jn->n", U, P[sym], U) for t, P in projectors.items() if sym in P}
            for n in range(E.numel()):
                states.append((float(E[n]), sym, dims.get(sym, 1), float(S2u[n]),
                               {cfg_names[k]: float(cw[k][n]) for k in range(len(sizes))},
                               {t: float(w[n]) for t, w in tw.items()}))
    states.sort(key=lambda s: s[0])
    e0 = states[0][0]

    levels: List[GroundLevel] = []
    group: List[tuple] = []

    def close():
        g = sum(x[2] for x in group)
        mean = lambda f: sum(f(x) * x[2] for x in group) / g
        confs = {k: mean(lambda x, k=k: x[4].get(k, 0.0)) for k in group[0][4]}
        terms = {k: mean(lambda x, k=k: x[5].get(k, 0.0)) for k in {t for x in group for t in x[5]}}
        s2m = mean(lambda x: x[3])
        levels.append(GroundLevel(
            energy=group[0][0], relative_meV=1000 * (group[0][0] - e0), degeneracy=g,
            irreps=sorted({x[1] for x in group}), S2=s2m, S_eff=_s_eff(s2m), configurations=confs,
            terms=dict(sorted(terms.items(), key=lambda kv: -kv[1])),
        ))

    for st in states:
        if group and st[0] - group[0][0] > degeneracy_tol:
            close()
            group = []
            if len(levels) >= n_levels:
                break
        group.append(st)
    if group and len(levels) < n_levels:
        close()

    # thermal averages over every state (normalised Boltzmann)
    kT = K_B_FLOAT * T if T > 0 else None
    w = [x[2] * (math.exp(-(x[0] - e0) / kT) if kT else float(x[0] - e0 <= degeneracy_tol)) for x in states]
    Z = sum(w)
    thermal_S2 = sum(wi * x[3] for wi, x in zip(w, states)) / Z
    thermal_conf = {k: sum(wi * x[4].get(k, 0.0) for wi, x in zip(w, states)) / Z for k in cfg_names}
    thermal_terms = {t: sum(wi * x[5].get(t, 0.0) for wi, x in zip(w, states)) / Z for t in projectors}
    thermal_terms = dict(sorted(thermal_terms.items(), key=lambda kv: -kv[1]))
    return GroundStateAnalysis(levels=levels, temperature=T, thermal_S2=thermal_S2,
                               thermal_S_eff=_s_eff(thermal_S2), thermal_configurations=thermal_conf,
                               thermal_terms=thermal_terms)
