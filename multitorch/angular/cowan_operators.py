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
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from multitorch.angular.rme import (
    _j_basis_for_terms,
    _lsterms_and_cfp,
    compute_coulomb_blocks,
    compute_double_tensor_ls,
    compute_soc_blocks,
    compute_two_shell_operators,
)
from multitorch.angular.wigner import wigner6j

Shell = Tuple[int, int]  # (l, n)


@dataclass(frozen=True)
class ConfigurationOperators:
    """Operators of one configuration in the Fortran store basis.

    ``blocks[name][J]`` is a (dim_J, dim_J) array carrying sqrt(2J+1) and
    relative to the configuration average. Names: ``F{k}_{ii}`` (intra-shell
    Coulomb), ``F{k}_12`` / ``G{k}_12`` (inter-shell direct / exchange),
    ``zeta_{i}`` (spin-orbit of shell i).
    """

    shells: Tuple[Shell, ...]
    dims: Dict[float, int]
    blocks: Dict[str, Dict[float, np.ndarray]]

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
    return ConfigurationOperators(((l, n),), {J: len(st) for J, st in basis.items()}, blocks)


def _two_shell(s1: Shell, s2: Shell) -> ConfigurationOperators:
    basis, ops = compute_two_shell_operators(*s1, *s2)
    g1, g2 = _term_gauge(*s1), _term_gauge(*s2)
    signs = {J: np.array([g1[s.term1_idx] * g2[s.term2_idx] for s in st]) for J, st in basis.items()}
    blocks = {name: _gauged(b, signs) for name, b in ops.items()}
    return ConfigurationOperators((s1, s2), {J: len(st) for J, st in basis.items()}, blocks)


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
    return ConfigurationOperators((s1, s2, s3), {J: len(st) for J, st in basis.items()}, blocks)


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
        return ConfigurationOperators((), {0.0: 1}, {})
    if len(shells) == 1:
        return _one_shell(*shells[0])
    if len(shells) == 2:
        return _two_shell(*shells)
    if len(shells) == 3:
        return _three_shell_spectator(*shells)
    raise NotImplementedError(f"{len(shells)} open shells")
