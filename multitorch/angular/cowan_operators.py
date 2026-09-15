"""
Parameter-free Hamiltonian operators in the Fortran (ttrcg) basis of a configuration.

Every HAMILTONIAN block that ttrcg writes into a ``.rme_rcg`` store is linear in
the atomic parameters of its configuration::

    H(J) = E_av·sqrt(2J+1)·I + Σ_k F^k·O_{F^k}(J) + Σ_k G^k·O_{G^k}(J) + Σ_i ζ_i·O_{ζ_i}(J)

This module returns the operators O(J) for a configuration given as its open
shells in Cowan order (the order of the ``%P06 D08 D10`` header lines), in the
same basis and phase convention as the Fortran store, so that a fixture block
can be decomposed exactly (``hamiltonian/build_cowan.py``).

Basis convention (verified against all bundled fixtures, d^0..d^9):

* State order within a J block is the order of
  :func:`~multitorch.angular.rme.build_two_shell_j_basis` (and its natural
  extension to three shells: generate over terms and intermediate (S12, L12),
  then stable-sort by (-S, -L)).
* Term phases differ from the CFP-derived basis of :mod:`~multitorch.angular.rme`
  by σ(αSL) = (-1)^(L + S - S_min), S_min = (n mod 2)/2, per shell. Our own
  operators (SHELL, SOC, MULTIPOLE) are mutually consistent; only the overall
  gauge differs, which is invisible to the Coulomb operator and to spectra but
  not to a block-by-block decomposition. The from-scratch pipeline keeps its
  own gauge.

Three open shells occur for the ligand-hole configurations of pyctm's LMCT
runs, e.g. 2p^5 3d^7 L^9 (``P05 D07 D09``). There the third shell carries no
parameters (no ζ_L, no F/G with L), so operators are generated for shells 1
and 2 only and lifted with shell 3 as a spectator. A fixture that did put
parameters on the third shell would fail the decomposition residual check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from multitorch.angular.rme import (
    LSTerm,
    _j_basis_for_terms,
    _lsterms_and_cfp,
    compute_coulomb_blocks,
    compute_double_tensor_ls,
    compute_soc_blocks,
    compute_two_shell_operators,
    compute_uk_ls,
    recpjp,
    uncpla,
    uncplb,
)
from multitorch.angular.wigner import wigner6j, wigner9j

Shell = Tuple[int, int]  # (l, n)


@dataclass(frozen=True)
class ConfigurationOperators:
    """Operators of one configuration in the Fortran store basis.

    ``blocks[name][J]`` is a (dim_J, dim_J) array carrying sqrt(2J+1) and
    relative to the configuration average. Names: ``F{k}_{ii}`` (intra-shell
    Coulomb), ``F{k}_12`` / ``G{k}_12`` (inter-shell direct / exchange),
    ``zeta_{i}`` (spin-orbit of shell i). ``states[J]`` lists the total
    ``(S, L)`` of every basis row, in block order.
    """

    shells: Tuple[Shell, ...]
    dims: Dict[float, int]
    blocks: Dict[str, Dict[float, np.ndarray]]
    states: Dict[float, List[Tuple[float, float]]] = field(default_factory=dict)

    @staticmethod
    def is_soc(name: str) -> bool:
        return name.startswith("zeta")


def _phase(x: float) -> float:
    return -1.0 if int(round(x)) % 2 else 1.0


def _term_gauge(l: int, n: int) -> Dict[int, float]:
    """σ(αSL) = (-1)^(L+S-S_min): Fortran term phase relative to the CFP basis."""
    terms, _, _ = _lsterms_and_cfp(l, n)
    s_min = (n % 2) / 2.0
    return {t.index: _phase(t.L + t.S - s_min) for t in terms}


def _triangle(a: float, b: float) -> List[float]:
    out, x = [], abs(a - b)
    while x <= a + b + 1e-9:
        out.append(x)
        x += 1.0
    return out


def _gauged(blocks: Dict[float, np.ndarray], signs: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    return {J: signs[J][:, None] * M * signs[J][None, :] for J, M in blocks.items()}


def _one_shell(l: int, n: int) -> ConfigurationOperators:
    terms, _, _ = _lsterms_and_cfp(l, n)
    basis = _j_basis_for_terms(terms)
    gauge = _term_gauge(l, n)
    signs = {J: np.array([gauge[s.ls_term.index] for s in st]) for J, st in basis.items()}
    blocks: Dict[str, Dict[float, np.ndarray]] = {}
    for (k, J), M in compute_coulomb_blocks(l, n).items():
        blocks.setdefault(f"F{k}_11", {})[J] = M
    blocks = {name: _gauged(b, signs) for name, b in blocks.items()}
    blocks["zeta_1"] = _gauged(compute_soc_blocks(l, n), signs)
    states = {J: [(float(s.ls_term.S), float(s.ls_term.L)) for s in st] for J, st in basis.items()}
    return ConfigurationOperators(((l, n),), {J: len(st) for J, st in basis.items()}, blocks, states)


def _two_shell(s1: Shell, s2: Shell) -> ConfigurationOperators:
    basis, ops = compute_two_shell_operators(*s1, *s2)
    g1, g2 = _term_gauge(*s1), _term_gauge(*s2)
    signs = {J: np.array([g1[s.term1_idx] * g2[s.term2_idx] for s in st]) for J, st in basis.items()}
    blocks = {name: _gauged(b, signs) for name, b in ops.items()}
    states = {J: [(float(s.S_total), float(s.L_total)) for s in st] for J, st in basis.items()}
    return ConfigurationOperators((s1, s2), {J: len(st) for J, st in basis.items()}, blocks, states)


def _three_shell_spectator(s1: Shell, s2: Shell, s3: Shell) -> ConfigurationOperators:
    """Operators of shells 1 and 2 in the ((1 2) S12 L12, 3) S L J basis; shell 3 is inert.

    Coulomb-type operators are scalars in the (1 2) spin and orbital spaces
    separately, so their elements are the two-shell LS elements, diagonal in
    (S12, L12, S, L) and in the shell-3 term. The spin-orbit operator of shell
    i is a (1 1) double tensor: two-shell doubly reduced element, lifted
    through the spectator coupling (Edmonds 7.1.7) and projected to J.
    """
    (l1, n1), (l2, n2), (l3, n3) = s1, s2, s3
    T1, _, _ = _lsterms_and_cfp(l1, n1)
    T2, _, _ = _lsterms_and_cfp(l2, n2)
    T3, _, _ = _lsterms_and_cfp(l3, n3)
    basis2, ops2 = compute_two_shell_operators(l1, n1, l2, n2)
    g1, g2, g3 = _term_gauge(*s1), _term_gauge(*s2), _term_gauge(*s3)

    # Pair (1 2) states and the position of each in the two-shell J = S12+L12 block,
    # where both members of any scalar-operator matrix element are present.
    pairs = [(a, b, S12, L12) for a in T1 for b in T2
             for S12 in _triangle(a.S, b.S) for L12 in _triangle(a.L, b.L)]
    pos = {}
    for J, st in basis2.items():
        for i, s in enumerate(st):
            pos[(s.term1_idx, s.term2_idx, s.S_total, s.L_total, J)] = i

    basis: Dict[float, List[Tuple[int, object, float, float]]] = {}
    for ip, (_, _, S12, L12) in enumerate(pairs):
        for c in T3:
            for S in _triangle(S12, c.S):
                for L in _triangle(L12, c.L):
                    for J in _triangle(L, S):
                        basis.setdefault(J, []).append((ip, c, S, L))
    for st in basis.values():
        st.sort(key=lambda x: (-x[2], -x[3]))

    V = {1: compute_double_tensor_ls(l1, n1, 1)[1], 2: compute_double_tensor_ls(l2, n2, 1)[1]}
    lfac = {1: math.sqrt(l1 * (l1 + 1) * (2 * l1 + 1)), 2: math.sqrt(l2 * (l2 + 1) * (2 * l2 + 1))}

    def pair_soc(p: int, q: int, shell: int) -> float:
        """<(a b) S12 L12 || V^(11)(shell) || (a' b') S12' L12'> incl. <l||l||l>."""
        a, b, S, L = pairs[p]
        ap, bp, Sp, Lp = pairs[q]
        if shell == 1:
            if b.index != bp.index:
                return 0.0
            rs = _phase(a.S + b.S + Sp + 1) * math.sqrt((2 * S + 1) * (2 * Sp + 1)) * wigner6j(a.S, S, b.S, Sp, ap.S, 1)
            rl = _phase(a.L + b.L + Lp + 1) * math.sqrt((2 * L + 1) * (2 * Lp + 1)) * wigner6j(a.L, L, b.L, Lp, ap.L, 1)
            return lfac[1] * rs * rl * V[1][a.index, ap.index]
        if a.index != ap.index:
            return 0.0
        rs = _phase(a.S + bp.S + S + 1) * math.sqrt((2 * S + 1) * (2 * Sp + 1)) * wigner6j(b.S, S, a.S, Sp, bp.S, 1)
        rl = _phase(a.L + bp.L + L + 1) * math.sqrt((2 * L + 1) * (2 * Lp + 1)) * wigner6j(b.L, L, a.L, Lp, bp.L, 1)
        return lfac[2] * rs * rl * V[2][b.index, bp.index]

    scalar_names = [name for name in ops2 if not name.startswith("zeta")]
    names = scalar_names + ["zeta_1", "zeta_2"]
    blocks: Dict[str, Dict[float, np.ndarray]] = {name: {} for name in names}
    for J, st in basis.items():
        d = len(st)
        w = math.sqrt(2 * J + 1)
        mats = {name: np.zeros((d, d)) for name in names}
        for i, (p, c, S, L) in enumerate(st):
            a, b, S12, L12 = pairs[p]
            for j, (q, cp, Sp, Lp) in enumerate(st):
                if c.index != cp.index:
                    continue
                ap, bp, S12p, L12p = pairs[q]
                if (S, L, S12, L12) == (Sp, Lp, S12p, L12p):
                    Js = S12 + L12
                    ii = pos[(a.index, b.index, S12, L12, Js)]
                    jj = pos[(ap.index, bp.index, S12, L12, Js)]
                    for name in scalar_names:
                        mats[name][i, j] = ops2[name][Js][ii, jj] / math.sqrt(2 * Js + 1) * w
                six = wigner6j(J, Lp, Sp, 1, S, L)
                if abs(six) < 1e-14:
                    continue
                lift = (_phase(S12 + c.S + Sp + 1) * math.sqrt((2 * S + 1) * (2 * Sp + 1))
                        * wigner6j(S12, S, c.S, Sp, S12p, 1)
                        * _phase(L12 + c.L + Lp + 1) * math.sqrt((2 * L + 1) * (2 * Lp + 1))
                        * wigner6j(L12, L, c.L, Lp, L12p, 1))
                if abs(lift) < 1e-14:
                    continue
                jfac = _phase(S + Lp + J) * six * w * lift
                for shell in (1, 2):
                    mats[f"zeta_{shell}"][i, j] = jfac * pair_soc(p, q, shell)
        signs = np.array([g1[pairs[p][0].index] * g2[pairs[p][1].index] * g3[c.index] for p, c, _, _ in st])
        for name in names:
            blocks[name][J] = signs[:, None] * mats[name] * signs[None, :]
    states = {J: [(float(S), float(L)) for _, _, S, L in st] for J, st in basis.items()}
    return ConfigurationOperators((s1, s2, s3), {J: len(st) for J, st in basis.items()}, blocks, states)


@lru_cache(maxsize=64)
def configuration_operators(open_shells: Tuple[Shell, ...]) -> ConfigurationOperators:
    """Operators for a configuration given by its open shells ``((l, n), ...)`` in Cowan order.

    Closed and empty shells must be removed by the caller; they do not enter
    the coupling. The returned arrays are shared (cached): do not mutate.
    """
    shells = tuple((int(l), int(n)) for l, n in open_shells)
    for l, n in shells:
        if not 0 < n < 4 * l + 2:
            raise ValueError(f"shell (l={l}, n={n}) is not open")
    if len(shells) == 0:
        return ConfigurationOperators((), {0.0: 1}, {}, {0.0: [(0.0, 0.0)]})
    if len(shells) == 1:
        return _one_shell(*shells[0])
    if len(shells) == 2:
        return _two_shell(*shells)
    if len(shells) == 3:
        return _three_shell_spectator(*shells)
    raise NotImplementedError(f"{len(shells)} open shells")


@lru_cache(maxsize=32)
def hopping_blocks(n_metal: int, rank: int, l: int = 2) -> Dict[Tuple[float, float], np.ndarray]:
    """Ligand-to-metal hopping operator blocks in the Fortran store basis (ground manifold).

    Bra: the metal configuration l^n (ligand shell closed); ket: the ligand-hole
    configuration l^(n+1) L^(4l+1) with the metal shell first, as ttrcg writes
    ``D n+1 ... D09``. The operator is the spin-scalar one-electron transfer of
    orbital rank ``rank`` (0, 2, 4 for d), which ttrcg stores as TRANSITION
    MULTIPOLE blocks and ttrac combines into eg/t2g (Oh) or b1/a1/b2/e (D4h)
    hybridisation channels.

    Built from the MUPOLE port (:func:`~multitorch.angular.rme.compute_multipole_blocks`
    with the ligand as a full "core" shell and a general rank) and brought to the
    store basis by (−1)^n σ(bra term) on rows and σ(metal term)·(−1)^(S+L) on
    columns. Oracle: every hopping block of the eight bundled Oh LMCT fixtures
    and nid8ct (tests/test_angular/test_hopping_operators.py), ≤ 2e-6.
    """
    from multitorch.angular.cfp import get_cfp_block
    from multitorch.angular.rme import build_two_shell_j_basis, compute_multipole_blocks

    terms_n, _, _ = _lsterms_and_cfp(l, n_metal)
    terms_n1, _, _ = _lsterms_and_cfp(l, n_metal + 1)
    terms_lig, _, _ = _lsterms_and_cfp(l, 4 * l + 1)
    parents = _lsterms_and_cfp(l, n_metal - 1)[0] if n_metal > 0 else []
    raw = compute_multipole_blocks(l, n_metal, l, 4 * l + 2, terms_n, parents,
                                   get_cfp_block(l, n_metal).cfp if n_metal > 0 else np.array([]), rank=rank)
    bra_basis = _j_basis_for_terms(terms_n)
    ket_basis = build_two_shell_j_basis(terms_lig, terms_n1)   # MUPOLE order: ligand first
    g_bra, g_met = _term_gauge(l, n_metal), _term_gauge(l, n_metal + 1)
    glob = _phase(n_metal)
    out: Dict[Tuple[float, float], np.ndarray] = {}
    for (Jb, Jk), M in raw.items():
        row = np.array([g_bra[s.ls_term.index] for s in bra_basis[Jb]])
        col = np.array([g_met[s.term2_idx] * _phase(s.S_total + s.L_total) for s in ket_basis[Jk]])
        out[(Jb, Jk)] = glob * row[:, None] * M * col[None, :]
    return out


def _store_basis(shells: Tuple[Shell, ...]) -> Dict[float, List[Tuple[tuple, Tuple[float, float], float, float]]]:
    """Fortran store basis of 1-3 open shells: ``[(terms, (S12, L12), S, L), ...]`` per J.

    ``terms`` holds one LSTerm per shell; ``(S12, L12)`` is the (1 2) intermediate
    coupling for three shells and ``None`` otherwise. Same order as the blocks of
    :func:`configuration_operators`.
    """
    T = [_lsterms_and_cfp(l, n)[0] for l, n in shells]
    if len(shells) == 1:
        return {J: [((s.ls_term,), None, s.ls_term.S, s.ls_term.L) for s in st]
                for J, st in _j_basis_for_terms(T[0]).items()}
    if len(shells) == 2:
        from multitorch.angular.rme import build_two_shell_j_basis
        return {J: [((T[0][s.term1_idx], T[1][s.term2_idx]), None, s.S_total, s.L_total) for s in st]
                for J, st in build_two_shell_j_basis(T[0], T[1]).items()}
    if len(shells) == 3:
        out: Dict[float, list] = {}
        for a in T[0]:
            for b in T[1]:
                for S12 in _triangle(a.S, b.S):
                    for L12 in _triangle(a.L, b.L):
                        for c in T[2]:
                            for S in _triangle(S12, c.S):
                                for L in _triangle(L12, c.L):
                                    for J in _triangle(L, S):
                                        out.setdefault(J, []).append(((a, b, c), (S12, L12), S, L))
        for st in out.values():
            st.sort(key=lambda x: (-x[2], -x[3]))
        return out
    raise NotImplementedError(f"{len(shells)} open shells")


def _orbital_lift(first: bool, L1: float, L1p: float, L2: float, L2p: float, L: float, Lp: float, k: int) -> float:
    """<(L1 L2) L || T^k || (L1' L2') L'> / <L_i || T^k || L_i'> for T acting on shell 1 or 2 (Edmonds 7.1.7/7.1.8)."""
    w = math.sqrt((2 * L + 1) * (2 * Lp + 1))
    if first:
        return _phase(L1 + L2 + Lp + k) * w * wigner6j(L1, L, L2, Lp, L1p, k)
    return _phase(L1 + L2p + L + k) * w * wigner6j(L2, L, L1, Lp, L2p, k)


@lru_cache(maxsize=64)
def shell_tensor_blocks(open_shells: Tuple[Shell, ...], shell: int, rank: int) -> Dict[Tuple[float, float], np.ndarray]:
    """Orbital unit tensor U^(rank) of one shell in the Fortran store basis (SHELL blocks).

    ``open_shells`` in Cowan order, ``shell`` the 0-based index of the shell the
    operator acts on (the metal d shell for crystal-field operators). Blocks are
    keyed ``(J_bra, J_ket)`` and carry the COWAN √((2J+1)(2J'+1)) (UNCPLA). The
    doubly reduced U^k of the shell (CFP) is lifted through the spectator
    couplings, J-projected, and brought to the store gauge by σ(term) of every
    shell on rows and columns; no block phase remains.

    Oracle: every SHELL block of the ground and final configurations of the Oh
    LMCT fixtures and nid8ct, sections 2 and 3 (1, 2 and 3 open shells, metal
    first or second) — tests/test_angular/test_shell_tensor_operators.py.
    """
    shells = tuple((int(l), int(n)) for l, n in open_shells)
    if len(shells) == 3 and shell == 2:
        raise NotImplementedError("tensor on the third (spectator) shell")
    l, n = shells[shell]
    terms, parents, cfp = _lsterms_and_cfp(l, n)
    U = compute_uk_ls(l, n, rank, terms, parents, cfp)
    basis = _store_basis(shells)
    gauges = [_term_gauge(*s) for s in shells]

    def ls_element(sb, sk) -> float:
        tb, tk = sb[0], sk[0]
        if len(shells) == 1:
            return U[tb[0].index, tk[0].index]
        if len(shells) == 2:
            spectator = 1 - shell
            if tb[spectator].index != tk[spectator].index:
                return 0.0
            return U[tb[shell].index, tk[shell].index] * _orbital_lift(
                shell == 0, tb[0].L, tk[0].L, tb[1].L, tk[1].L, sb[3], sk[3], rank)
        (S12, L12), (S12p, L12p) = sb[1], sk[1]
        spectator = 1 - shell
        if tb[spectator].index != tk[spectator].index or tb[2].index != tk[2].index or abs(S12 - S12p) > 1e-9:
            return 0.0
        inner = U[tb[shell].index, tk[shell].index] * _orbital_lift(
            shell == 0, tb[0].L, tk[0].L, tb[1].L, tk[1].L, L12, L12p, rank)
        return inner * _orbital_lift(True, L12, L12p, tb[2].L, tk[2].L, sb[3], sk[3], rank)

    out: Dict[Tuple[float, float], np.ndarray] = {}
    for Jb, B in basis.items():
        rows = np.array([math.prod(g[t.index] for g, t in zip(gauges, s[0])) for s in B])
        for Jk, K in basis.items():
            if abs(Jb - Jk) > rank or Jb + Jk < rank:
                continue
            M = np.zeros((len(B), len(K)))
            for i, sb in enumerate(B):
                for j, sk in enumerate(K):
                    if abs(sb[2] - sk[2]) > 1e-9:
                        continue
                    ls = ls_element(sb, sk)
                    if abs(ls) < 1e-14:
                        continue
                    M[i, j] = uncpla(sb[3], sb[2], Jb, rank, sk[3], Jk) * ls
            cols = np.array([math.prod(g[t.index] for g, t in zip(gauges, s[0])) for s in K])
            out[(Jb, Jk)] = rows[:, None] * M * cols[None, :]
    return out


def _transfer_ls(m: int, R: int, t: object, lig: object, met: object, S_ml: float, L_ml: float,
                 l: int = 2, l_from: int = None) -> float:
    """MUPOLE LS element <l^m t; l'^(4l'+2) || T^(0R) || (l'^(4l'+1) lig, l^(m+1) met) S_ml L_ml>.

    The LS core of :func:`~multitorch.angular.rme.compute_multipole_blocks`: one
    electron moves from the full shell l' (``l_from``, default l: the ligand;
    1 for the 2p core) into l^m. Source-shell-first coupling, no J projection,
    no MULTIPOLE block phase. Kept separate so that it can be recoupled under
    spectator shells (core hole, ligand hole).
    """
    from multitorch.angular.cfp import get_cfp_block

    if abs(t.S - S_ml) > 1e-9 or abs(t.L - L_ml) > R + 1e-9 or t.L + L_ml < R - 1e-9:
        return 0.0
    l_from = l if l_from is None else l_from
    n_lig, n_met = 4 * l_from + 2, m + 1
    tc = math.sqrt(n_lig * n_met) * (-1.0 if n_met % 2 == 0 else 1.0) * _phase(t.L + L_ml + 1)
    lig_cfp = get_cfp_block(l_from, n_lig).cfp
    if lig_cfp is not None and lig_cfp.size:
        tc *= lig_cfp[0, lig.index]
    met_cfp = get_cfp_block(l, n_met).cfp
    if met_cfp is not None and met_cfp.size:
        tc *= met_cfp[met.index, t.index]
    if abs(t.S) > 1e-10:
        tc *= recpjp(lig.S, 0.5, 0.0, t.S, t.S, met.S)
    if abs(t.L) > 1e-10:
        tc *= _phase(l + t.L - met.L) * math.sqrt((2 * met.L + 1) * (2 * t.L + 1) * (2 * L_ml + 1))
        tc *= wigner9j(0.0, lig.L, l_from, t.L, met.L, l, t.L, L_ml, R)
    elif lig.L > 1e-10:
        tc *= uncplb(lig.L, l_from, t.L, R, l, L_ml)
    return tc


@lru_cache(maxsize=32)
def final_state_hopping_blocks(n_metal: int, rank: int, l: int = 2, l_core: int = 1) -> Dict[Tuple[float, float], np.ndarray]:
    """Ligand-to-metal hopping under a core hole, in the Fortran store basis (final manifold).

    Bra: 2p^5 d^(n+1) (``P05 D n+1``); ket: the ligand-hole configuration
    2p^5 d^(n+2) L^9 (``P05 D n+2 D09``), or 2p^5 L^9 when the metal shell
    closes (d^8). ``n_metal`` is the ground-state metal occupation n, as for
    :func:`hopping_blocks`; ``rank`` 0, 2, 4.

    The ket ((core metal') S12 L12, ligand) S L is recoupled to (core, (metal'
    ligand) S_ml L_ml) S L (spin and orbit 6j); the transfer then acts on the
    second member of (core, metal) with the core a spectator (Edmonds 7.1.8),
    on the MUPOLE LS element :func:`_transfer_ls`, and is J-projected (UNCPLA).
    Store gauge: σ(core)σ(metal) on rows, σ(core)σ(metal')σ(ligand) on columns,
    and an overall −1.

    Oracle: every section-3 TRANSITION MULTIPOLE block of the eight bundled Oh
    LMCT fixtures and nid8ct (tests/test_angular/test_hopping_operators.py).
    """
    n_lig = 4 * l + 2
    mb = n_metal + 1
    if not 0 < mb < n_lig:
        raise ValueError(f"final-state hopping needs an open metal shell in the bra, got d^{mb}")
    n_core = 4 * l_core + 1
    core_terms = _lsterms_and_cfp(l_core, n_core)[0]
    met_b = _lsterms_and_cfp(l, mb)[0]
    closed = mb + 1 == n_lig
    if closed:
        met_k = [LSTerm(index=0, S=0.0, L=0.0, seniority=0, label="1S")]
        g_mk = {0: 1.0}
    else:
        met_k = _lsterms_and_cfp(l, mb + 1)[0]
        g_mk = _term_gauge(l, mb + 1)
    lig_terms = _lsterms_and_cfp(l, n_lig - 1)[0]
    g_core, g_mb, g_lig = _term_gauge(l_core, n_core), _term_gauge(l, mb), _term_gauge(l, n_lig - 1)

    bra_shells = ((l_core, n_core), (l, mb))
    ket_shells = ((l_core, n_core), (l, n_lig - 1)) if closed else ((l_core, n_core), (l, mb + 1), (l, n_lig - 1))
    bra = _store_basis(bra_shells)
    ket: Dict[float, list] = {}
    for J, st in _store_basis(ket_shells).items():
        if closed:   # (core, ligand) S L is ((core, 1S) S_c L_c, ligand) S L
            ket[J] = [(c, met_k[0], c.S, c.L, lam, S, L) for (c, lam), _, S, L in st]
        else:
            ket[J] = [(c, m, S12, L12, lam, S, L) for (c, m, lam), (S12, L12), S, L in st]

    R = int(rank)
    out: Dict[Tuple[float, float], np.ndarray] = {}
    for Jb, B in bra.items():
        rows = np.array([g_core[c.index] * g_mb[t.index] for (c, t), _, _, _ in B])
        for Jk, K in ket.items():
            if abs(Jb - Jk) > R or Jb + Jk < R:
                continue
            M = np.zeros((len(B), len(K)))
            for i, ((c, t), _, S, L) in enumerate(B):
                for j, (a, b, S12, L12, lam, Sk, Lk) in enumerate(K):
                    if a.index != c.index or abs(S - Sk) > 1e-9 or t.S not in _triangle(b.S, lam.S):
                        continue
                    S_ml = t.S
                    ws = (_phase(c.S + b.S + lam.S + S) * math.sqrt((2 * S12 + 1) * (2 * S_ml + 1))
                          * wigner6j(c.S, b.S, S12, lam.S, S, S_ml))
                    if abs(ws) < 1e-14:
                        continue
                    tot = 0.0
                    for L_ml in _triangle(b.L, lam.L):
                        h = _transfer_ls(mb, R, t, lam, b, S_ml, L_ml, l)
                        if abs(h) < 1e-14:
                            continue
                        wl = (_phase(c.L + b.L + lam.L + Lk) * math.sqrt((2 * L12 + 1) * (2 * L_ml + 1))
                              * wigner6j(c.L, b.L, L12, lam.L, Lk, L_ml))
                        spec = (_phase(c.L + L_ml + L + R) * math.sqrt((2 * L + 1) * (2 * Lk + 1))
                                * wigner6j(t.L, L, c.L, Lk, L_ml, R))
                        tot += wl * spec * h
                    M[i, j] = uncpla(L, S, Jb, R, Lk, Jk) * ws * tot
            cols = np.array([g_core[a.index] * g_mk[b.index] * g_lig[lam.index] for a, b, _, _, lam, _, _ in K])
            out[(Jb, Jk)] = -rows[:, None] * M * cols[None, :]
    return out


@lru_cache(maxsize=32)
def dipole_blocks(n_metal: int, ligand_hole: bool = False, l: int = 2, l_core: int = 1) -> Dict[Tuple[float, float], np.ndarray]:
    """2p → 3d dipole (rank 1) blocks in the Fortran store basis.

    ``ligand_hole=False``: bra d^n (``D n``), ket 2p^5 d^(n+1) (``P05 D n+1``),
    the store's section 0. ``ligand_hole=True``: bra d^(n+1) L^9, ket
    2p^5 d^(n+2) L^9 (2p^5 L^9 when the metal closes), section 1, with the
    ligand hole a spectator (Edmonds 7.1.7 on the (core, metal) pair).

    The MUPOLE LS element :func:`_transfer_ls` is J-projected (UNCPLA) and
    brought to the store gauge by σ on rows, σ times the (core, metal) coupling
    swap (−1)^(S_c+S_m−S_cm + L_c+L_m−L_cm) on columns, and (−1)^m for m metal
    electrons in the bra.

    Oracle: every MULTIPOLE block of sections 0 and 1 of the eight bundled
    two-configuration fixtures (tests/test_angular/test_ct_operators.py).
    """
    m = n_metal + int(ligand_hole)
    n_core, n_lig = 4 * l_core + 1, 4 * l + 1
    if not 0 < m < 4 * l + 2:
        raise ValueError(f"dipole blocks need an open metal shell in the bra, got d^{m}")
    closed = m + 1 == 4 * l + 2
    one_s = LSTerm(index=0, S=0.0, L=0.0, seniority=0, label="1S")
    if ligand_hole:
        bra_shells = ((l, m), (l, n_lig))
        ket_shells = ((l_core, n_core), (l, n_lig)) if closed else ((l_core, n_core), (l, m + 1), (l, n_lig))
    else:
        bra_shells = ((l, m),)
        ket_shells = ((l_core, n_core),) if closed else ((l_core, n_core), (l, m + 1))
    gauges = {sh: (_term_gauge(*sh) if sh[1] < 4 * sh[0] + 2 else {0: 1.0}) for sh in set(bra_shells + ket_shells)}

    def unpack_ket(terms, i12, S, L):
        """(core, metal', ligand or None, S_cm, L_cm) of a ket state."""
        if not ligand_hole:
            c, mp = (terms[0], one_s) if closed else terms
            return c, mp, None, S, L
        if closed:
            c, lam = terms
            return c, one_s, lam, c.S, c.L
        c, mp, lam = terms
        return c, mp, lam, i12[0], i12[1]

    bra, ket = _store_basis(bra_shells), _store_basis(ket_shells)
    glob = _phase(m)
    out: Dict[Tuple[float, float], np.ndarray] = {}
    for Jb, B in bra.items():
        rows = np.array([math.prod(gauges[sh][t.index] for sh, t in zip(bra_shells, terms)) for terms, _, _, _ in B])
        for Jk, K in ket.items():
            if abs(Jb - Jk) > 1 or Jb + Jk < 1:
                continue
            kets = [unpack_ket(*s) for s in K]
            M = np.zeros((len(B), len(K)))
            for i, (terms, _, S, L) in enumerate(B):
                t = terms[0]
                lam = terms[1] if ligand_hole else None
                for j, ((_, _, Sk, Lk), (c, mp, lamk, S_cm, L_cm)) in enumerate(zip(K, kets)):
                    if abs(S - Sk) > 1e-9:
                        continue
                    if ligand_hole:
                        if lamk.index != lam.index:
                            continue
                        ls = _transfer_ls(m, 1, t, c, mp, S_cm, L_cm, l, l_core)
                        if abs(ls) < 1e-14:
                            continue
                        ls *= _orbital_lift(True, t.L, L_cm, lam.L, lam.L, L, Lk, 1)
                    else:
                        ls = _transfer_ls(m, 1, t, c, mp, Sk, Lk, l, l_core)
                    if abs(ls) < 1e-14:
                        continue
                    M[i, j] = uncpla(L, S, Jb, 1, Lk, Jk) * ls
            cols = np.array([
                math.prod(gauges[sh][tt.index] for sh, tt in zip(ket_shells, terms))
                * _phase(c.S + mp.S - S_cm + c.L + mp.L - L_cm)
                for (terms, _, _, _), (c, mp, _, S_cm, L_cm) in zip(K, kets)])
            out[(Jb, Jk)] = glob * rows[:, None] * M * cols[None, :]
    return out
