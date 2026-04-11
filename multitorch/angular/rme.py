"""
Reduced matrix element (RME) builder.

Computes matrix elements of physical operators in the coupled multi-electron
basis using:
  1. Wigner-Eckart theorem
  2. Coefficients of fractional parentage (CFP)
  3. Recoupling via Wigner 6j symbols

The key formula for the unit tensor U^(k) in a configuration l^n:
  <l^n αSL || U^(k) || l^n α'S'L'> =
    n × δ_{SS'} × Σ_{parent ᾱS̄L̄}
      cfp(l^n αSL | l^(n-1) ᾱS̄L̄) × cfp(l^n α'S'L' | l^(n-1) ᾱS̄L̄)
      × (-1)^{L̄+l+L'+k} × √((2L+1)(2L'+1)) × {L k L'; l L̄ l}

The SHELL block values in the J-coupled basis (.rme_rcg format) are:
  SHELL(αSL;J, α'S'L';J', k) = UNCPLA(L,S,J,k,L',J') × U^(k)(αSL, α'S'L')

where UNCPLA(L,S,J,k,L',J') = (-1)^{L+S+J'+k} × √((2J+1)(2J'+1)) × {L J S; J' L' k}

This module computes both U^(k) from scratch (using CFP tables + Wigner 6j)
and the full SHELL block values including LS→J recoupling.

Reference:
  Cowan (1981) The Theory of Atomic Structure and Spectra, Ch. 9.
  de Groot & Kotani (2008) Core Level Spectroscopy of Solids.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from multitorch._constants import DTYPE
from multitorch.angular.wigner import wigner3j, wigner6j, wigner9j, clebsch_gordan


# ─────────────────────────────────────────────────────────────
# LS term and J-basis data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class LSTerm:
    """One LS-coupled term of an l^n configuration."""
    index: int       # 0-based index in the term list
    S: float         # spin quantum number
    L: float         # orbital angular momentum
    seniority: int   # seniority number (distinguishes repeated terms)
    label: str       # e.g., '1S', '3P', '1D', '3F', '1G'


@dataclass
class JBasisState:
    """One state in the J-coupled basis."""
    J: float         # total angular momentum
    ls_term: LSTerm  # parent LS term
    idx_in_J: int    # index within the J block


# ─────────────────────────────────────────────────────────────
# UNCPLA recoupling coefficient (Fortran ttrcg.f line 2141)
# ─────────────────────────────────────────────────────────────

def uncpla(L: float, S: float, J: float,
           k: float, Lp: float, Jp: float) -> float:
    """
    Recoupling coefficient for LS→J transformation of a tensor operator.

    UNCPLA(L, S, J, k, L', J') = (-1)^{L+S+J'+k} × √((2J+1)(2J'+1)) × {L J S; J' L' k}

    This is the Fortran UNCPLA function from ttrcg.f.
    """
    phase_exp = L + S + Jp + k
    phase = (-1.0) ** int(round(phase_exp))
    factor = math.sqrt((2.0 * J + 1.0) * (2.0 * Jp + 1.0))
    sixj = wigner6j(L, J, S, Jp, Lp, k)
    return phase * factor * sixj


# ─────────────────────────────────────────────────────────────
# Unit tensor U^(k) computation from CFP + 6j
# ─────────────────────────────────────────────────────────────

def compute_uk_ls(
    l: int,
    n: int,
    k: int,
    terms: List[LSTerm],
    parent_terms: List[LSTerm],
    cfp_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the unit tensor U^(k) matrix in the LS basis from CFP + 6j.

    <l^n αSL || U^(k) || l^n α'S'L'> =
      n × δ_{SS'} × Σ_{parent idx p}
        cfp[α_idx, p] × cfp[α'_idx, p]
        × (-1)^{Lp + l + L' + k} × √((2L+1)(2L'+1)) × {L k L'; l Lp l}

    Parameters
    ----------
    l : int
        Single-electron orbital quantum number (2 for d).
    n : int
        Number of electrons.
    k : int
        Tensor rank (0, 2, 4 for Coulomb; 1 for dipole).
    terms : list of LSTerm
        LS terms of l^n configuration (ordered as in CFP file).
    parent_terms : list of LSTerm
        LS terms of l^(n-1) configuration (ordered as in CFP file).
    cfp_matrix : np.ndarray, shape (n_terms, n_parent_terms)
        CFP values from rcg_cfp72.

    Returns
    -------
    np.ndarray, shape (n_terms, n_terms)
        U^(k) matrix in the LS term basis.
    """
    n_terms = len(terms)
    n_parents = len(parent_terms)
    uk = np.zeros((n_terms, n_terms), dtype=np.float64)

    fl = float(l)

    for i, ti in enumerate(terms):
        for j, tj in enumerate(terms):
            # Selection rule: δ_{SS'}
            if abs(ti.S - tj.S) > 1e-10:
                continue

            # Triangle rule: |L - L'| ≤ k ≤ L + L'
            if k > ti.L + tj.L or k < abs(ti.L - tj.L):
                continue

            val = 0.0
            for p, tp in enumerate(parent_terms):
                # CFP values
                cfp_i = cfp_matrix[i, p] if i < cfp_matrix.shape[0] and p < cfp_matrix.shape[1] else 0.0
                cfp_j = cfp_matrix[j, p] if j < cfp_matrix.shape[0] and p < cfp_matrix.shape[1] else 0.0

                if abs(cfp_i) < 1e-15 or abs(cfp_j) < 1e-15:
                    continue

                # 6j symbol {L k L'; l Lp l}
                sixj = wigner6j(ti.L, k, tj.L, fl, tp.L, fl)
                if abs(sixj) < 1e-15:
                    continue

                # Phase: (-1)^{Lp + l + L' + k}
                phase_exp = int(round(tp.L + fl + tj.L + k))
                phase = (-1.0) ** phase_exp

                val += cfp_i * cfp_j * phase * sixj

            # Factor: n × √((2L+1)(2L'+1))
            val *= n * math.sqrt((2.0 * ti.L + 1.0) * (2.0 * tj.L + 1.0))
            uk[i, j] = val

    return uk


# ─────────────────────────────────────────────────────────────
# J-basis SHELL block builder
# ─────────────────────────────────────────────────────────────

def build_j_basis(terms: List[LSTerm], J_min: float, J_max: float
                  ) -> Dict[float, List[JBasisState]]:
    """
    Build the J-coupled basis from LS terms.

    For each J value, collect all LS terms where |L-S| ≤ J ≤ L+S.

    Returns
    -------
    dict mapping J → list of JBasisState (ordered by term index)
    """
    j_basis: Dict[float, List[JBasisState]] = {}

    for J in _frange(J_min, J_max):
        states = []
        for term in terms:
            if abs(term.L - term.S) <= J + 1e-10 and J <= term.L + term.S + 1e-10:
                states.append(JBasisState(J=J, ls_term=term, idx_in_J=len(states)))
            # Check for integer/half-integer consistency
        if states:
            j_basis[J] = states

    return j_basis


def _frange(start: float, stop: float) -> List[float]:
    """Generate float range in steps of 1 from start to stop (inclusive)."""
    vals = []
    v = start
    while v <= stop + 0.01:
        vals.append(v)
        v += 1.0
    return vals


def compute_shell_blocks(
    l: int,
    n: int,
    k: int,
    terms: List[LSTerm],
    uk_ls: np.ndarray,
) -> Dict[Tuple[float, float], np.ndarray]:
    """
    Compute SHELL block RME values in the J-coupled basis.

    For each pair of J values (J_bra, J_ket) where |J_bra - J_ket| ≤ k ≤ J_bra + J_ket:
      SHELL(i,j) = UNCPLA(L_i, S_i, J_bra, k, L_j, J_ket) × U^(k)(term_i, term_j)

    Parameters
    ----------
    l : int
        Single-electron orbital quantum number.
    n : int
        Number of electrons.
    k : int
        Tensor rank.
    terms : list of LSTerm
        LS terms of l^n.
    uk_ls : np.ndarray
        U^(k) matrix in LS basis, shape (n_terms, n_terms).

    Returns
    -------
    dict mapping (J_bra, J_ket) → np.ndarray
        One matrix per (J_bra, J_ket) pair, shape (n_states_at_J_bra, n_states_at_J_ket).
    """
    # Determine J range
    S_max = max(t.S for t in terms)
    L_max = max(t.L for t in terms)
    J_min = 0.0 if (2 * S_max) % 2 == 0 else 0.5
    J_max = S_max + L_max

    j_basis = build_j_basis(terms, J_min, J_max)
    blocks: Dict[Tuple[float, float], np.ndarray] = {}

    for J_bra, states_bra in j_basis.items():
        for J_ket, states_ket in j_basis.items():
            # Triangle rule for operator rank k
            if k > J_bra + J_ket + 1e-10 or k < abs(J_bra - J_ket) - 1e-10:
                continue

            n_bra = len(states_bra)
            n_ket = len(states_ket)
            mat = np.zeros((n_bra, n_ket), dtype=np.float64)

            for ib, sb in enumerate(states_bra):
                for ik, sk in enumerate(states_ket):
                    ti = sb.ls_term
                    tj = sk.ls_term

                    # Get U^(k) value in LS basis
                    uk_val = uk_ls[ti.index, tj.index]
                    if abs(uk_val) < 1e-15:
                        continue

                    # UNCPLA recoupling
                    rc = uncpla(ti.L, ti.S, J_bra, k, tj.L, J_ket)
                    mat[ib, ik] = rc * uk_val

            # Only store non-zero blocks
            if np.any(np.abs(mat) > 1e-15):
                blocks[(J_bra, J_ket)] = mat

    return blocks


# ─────────────────────────────────────────────────────────────
# High-level: compute all SHELL blocks for a configuration
# ─────────────────────────────────────────────────────────────

def compute_all_shell_blocks(
    l: int,
    n: int,
    binpath: Optional[str] = None,
) -> Dict[Tuple[int, float, float], np.ndarray]:
    """
    Compute all SHELL block RME values for l^n from scratch.

    Uses CFP from rcg_cfp72 + Wigner 6j symbols to compute U^(k),
    then UNCPLA recoupling to get J-basis values.

    Parameters
    ----------
    l : int
        Single-electron orbital quantum number (2 for d).
    n : int
        Number of electrons.
    binpath : str or None
        Path to ttmult/bin/ directory.

    Returns
    -------
    dict mapping (k, J_bra, J_ket) → np.ndarray
        SHELL block matrices for all tensor ranks k and J pairs.
    """
    from multitorch.angular.cfp import get_cfp_block

    block = get_cfp_block(l, n, binpath)
    if block is None:
        raise ValueError(f"No CFP data for l={l}, n={n}")

    parent_block = get_cfp_block(l, n - 1, binpath)

    # Build term lists
    terms = [LSTerm(
        index=t.index, S=t.S, L=t.L,
        seniority=t.seniority,
        label=f"{int(2*t.S+1)}{t.L_label}",
    ) for t in block.terms]

    parent_terms = []
    if parent_block is not None and parent_block.terms:
        parent_terms = [LSTerm(
            index=t.index, S=t.S, L=t.L,
            seniority=t.seniority,
            label=f"{int(2*t.S+1)}{t.L_label}",
        ) for t in parent_block.terms]
    elif block.parent_terms:
        parent_terms = [LSTerm(
            index=t.index, S=t.S, L=t.L,
            seniority=t.seniority,
            label=f"{int(2*t.S+1)}{t.L_label}",
        ) for t in block.parent_terms]

    if block.cfp is None:
        raise ValueError(f"No CFP matrix for l={l}, n={n}")

    cfp = block.cfp

    all_blocks: Dict[Tuple[int, float, float], np.ndarray] = {}

    # Compute for each even k (Coulomb interaction)
    for k in range(0, 2 * l + 1, 2):
        uk_ls = compute_uk_ls(l, n, k, terms, parent_terms, cfp)
        shell_blocks = compute_shell_blocks(l, n, k, terms, uk_ls)
        for (J_bra, J_ket), mat in shell_blocks.items():
            all_blocks[(k, J_bra, J_ket)] = mat

    return all_blocks


# ─────────────────────────────────────────────────────────────
# SPIN operator: S_shell (rank-1 spin tensor)
# ─────────────────────────────────────────────────────────────

def uncplb(L: float, S: float, J: float,
           k: float, Sp: float, Jp: float) -> float:
    """
    Recoupling coefficient UNCPLB from ttrcg.f line 2151.

    UNCPLB(L, S, J, k, S', J') = (-1)^{L+J+S'+k} × √((2J+1)(2J'+1)) × {S J L; J' S' k}
    """
    phase_exp = L + J + Sp + k
    phase = (-1.0) ** int(round(phase_exp))
    factor = math.sqrt((2.0 * J + 1.0) * (2.0 * Jp + 1.0))
    sixj = wigner6j(S, J, L, Jp, Sp, k)
    return phase * factor * sixj


def compute_spin_blocks(
    terms: List[LSTerm],
) -> Dict[Tuple[float, float], np.ndarray]:
    """
    Compute SPIN1 block RME values in the J-coupled basis.

    For the spin operator S acting on a single shell:
      SPIN(αSL;J, α'S'L';J') = δ_{LL'} δ_{αα'} × UNCPLB(L,S,J,1,S',J') × √(S(S+1)(2S+1))

    where the α, L matching is enforced by selection rules.

    Parameters
    ----------
    terms : list of LSTerm
        LS terms of l^n.

    Returns
    -------
    dict mapping (J_bra, J_ket) → np.ndarray
    """
    k = 1  # spin is rank 1
    S_max = max(t.S for t in terms)
    L_max = max(t.L for t in terms)
    J_min = 0.0 if (2 * S_max) % 2 == 0 else 0.5
    J_max = S_max + L_max

    j_basis = build_j_basis(terms, J_min, J_max)
    blocks: Dict[Tuple[float, float], np.ndarray] = {}

    for J_bra, states_bra in j_basis.items():
        for J_ket, states_ket in j_basis.items():
            if k > J_bra + J_ket + 1e-10 or k < abs(J_bra - J_ket) - 1e-10:
                continue

            n_bra = len(states_bra)
            n_ket = len(states_ket)
            mat = np.zeros((n_bra, n_ket), dtype=np.float64)

            for ib, sb in enumerate(states_bra):
                for ik, sk in enumerate(states_ket):
                    ti = sb.ls_term
                    tj = sk.ls_term

                    # Selection rules: same L, same seniority (same parent term)
                    if abs(ti.L - tj.L) > 1e-10:
                        continue
                    if ti.seniority != tj.seniority:
                        continue

                    # Spin triangle: |S - S'| ≤ 1 ≤ S + S'
                    if 1 > ti.S + tj.S + 1e-10:
                        continue
                    if 1 < abs(ti.S - tj.S) - 1e-10:
                        continue

                    # UNCPLB recoupling
                    rc = uncplb(ti.L, ti.S, J_bra, k, tj.S, J_ket)
                    # Spin reduced matrix element: √(S(S+1)(2S+1))
                    Si = ti.S
                    spin_rme = math.sqrt(Si * (Si + 1.0) * (2.0 * Si + 1.0))
                    mat[ib, ik] = rc * spin_rme

            if np.any(np.abs(mat) > 1e-15):
                blocks[(J_bra, J_ket)] = mat

    return blocks


# ─────────────────────────────────────────────────────────────
# ORBIT operator: L_shell (rank-1 orbital angular momentum tensor)
# ─────────────────────────────────────────────────────────────

def compute_orbit_blocks(
    terms: List[LSTerm],
) -> Dict[Tuple[float, float], np.ndarray]:
    """
    Compute ORBIT block RME values in the J-coupled basis.

    For the orbital angular momentum operator L acting on a single shell:
      ORBIT(αSL;J, α'S'L';J') = δ_{term} × UNCPLA(L, S, J, 1, L, J') × √(L(L+1)(2L+1))

    Selection rules: same term (same α, S, L). Only J coupling differs.

    Parameters
    ----------
    terms : list of LSTerm
        LS terms of l^n.

    Returns
    -------
    dict mapping (J_bra, J_ket) → np.ndarray
    """
    k = 1  # orbital angular momentum is rank 1
    S_max = max(t.S for t in terms)
    L_max = max(t.L for t in terms)
    J_min = 0.0 if (2 * S_max) % 2 == 0 else 0.5
    J_max = S_max + L_max

    j_basis = build_j_basis(terms, J_min, J_max)
    blocks: Dict[Tuple[float, float], np.ndarray] = {}

    for J_bra, states_bra in j_basis.items():
        for J_ket, states_ket in j_basis.items():
            if k > J_bra + J_ket + 1e-10 or k < abs(J_bra - J_ket) - 1e-10:
                continue

            n_bra = len(states_bra)
            n_ket = len(states_ket)
            mat = np.zeros((n_bra, n_ket), dtype=np.float64)

            for ib, sb in enumerate(states_bra):
                for ik, sk in enumerate(states_ket):
                    ti = sb.ls_term
                    tj = sk.ls_term

                    # Selection rule: same term (same α, S, L, seniority)
                    if ti.index != tj.index:
                        continue

                    # UNCPLA recoupling (same as SHELL but with k=1)
                    rc = uncpla(ti.L, ti.S, J_bra, k, ti.L, J_ket)
                    # Orbital reduced matrix element: √(L(L+1)(2L+1))
                    Li = ti.L
                    orbit_rme = math.sqrt(Li * (Li + 1.0) * (2.0 * Li + 1.0))
                    mat[ib, ik] = rc * orbit_rme

            if np.any(np.abs(mat) > 1e-15):
                blocks[(J_bra, J_ket)] = mat

    return blocks


# ─────────────────────────────────────────────────────────────
# Recoupling coefficients for multi-shell operators
# ─────────────────────────────────────────────────────────────

def recpsh(j1: float, j2: float, jp: float,
           j3: float, j: float, jpp: float) -> float:
    """RECPSH from ttrcg.f line 2161."""
    phase = (-1.0) ** int(round(j1 + j2 + j3 + j))
    factor = math.sqrt((2.0 * jp + 1.0) * (2.0 * jpp + 1.0))
    return phase * factor * wigner6j(j1, j2, jp, j3, j, jpp)


def recpjp(j1: float, j2: float, jp: float,
           j3: float, j: float, jpp: float) -> float:
    """RECPJP from ttrcg.f line 2172."""
    phase = (-1.0) ** int(round(j1 + j + jpp))
    factor = math.sqrt((2.0 * jp + 1.0) * (2.0 * jpp + 1.0))
    return phase * factor * wigner6j(j1, j2, jp, j3, j, jpp)


def recpex(j1: float, j2: float, jp: float,
           j3: float, j: float, jpp: float) -> float:
    """RECPEX from ttrcg.f line 2183."""
    phase = (-1.0) ** int(round(j2 + j3 + jp + jpp))
    factor = math.sqrt((2.0 * jp + 1.0) * (2.0 * jpp + 1.0))
    return phase * factor * wigner6j(j2, j1, jp, j3, j, jpp)


# ─────────────────────────────────────────────────────────────
# Two-shell coupled basis for excited states
# ─────────────────────────────────────────────────────────────

@dataclass
class TwoShellState:
    """One state in the coupled basis of two open shells."""
    S1: float     # spin of shell 1
    L1: float     # orbital AM of shell 1
    S2: float     # spin of shell 2
    L2: float     # orbital AM of shell 2
    S_total: float  # coupled total spin
    L_total: float  # coupled total orbital AM
    J: float       # total angular momentum
    term1_idx: int  # term index within shell 1
    term2_idx: int  # term index within shell 2
    idx_in_J: int   # index within the J block


def build_two_shell_j_basis(
    terms1: List[LSTerm],
    terms2: List[LSTerm],
) -> Dict[float, List[TwoShellState]]:
    """
    Build J-coupled basis for two open shells.

    States: |(α1 S1 L1)(α2 S2 L2) S_total L_total; J⟩

    Parameters
    ----------
    terms1 : list of LSTerm
        LS terms of shell 1.
    terms2 : list of LSTerm
        LS terms of shell 2.

    Returns
    -------
    dict mapping J → list of TwoShellState
    """
    j_basis: Dict[float, List[TwoShellState]] = {}

    # Generate all coupled states
    for t1 in terms1:
        for t2 in terms2:
            # Couple spins: |S1 - S2| ≤ S_total ≤ S1 + S2
            S_min = abs(t1.S - t2.S)
            S_max = t1.S + t2.S
            S_val = S_min
            while S_val <= S_max + 1e-10:
                # Couple orbital: |L1 - L2| ≤ L_total ≤ L1 + L2
                L_min = abs(t1.L - t2.L)
                L_max = t1.L + t2.L
                L_val = L_min
                while L_val <= L_max + 1e-10:
                    # Couple to J: |L_total - S_total| ≤ J ≤ L_total + S_total
                    J_min = abs(L_val - S_val)
                    J_max = L_val + S_val
                    J_val = J_min
                    while J_val <= J_max + 1e-10:
                        if J_val not in j_basis:
                            j_basis[J_val] = []
                        state = TwoShellState(
                            S1=t1.S, L1=t1.L,
                            S2=t2.S, L2=t2.L,
                            S_total=S_val, L_total=L_val,
                            J=J_val,
                            term1_idx=t1.index,
                            term2_idx=t2.index,
                            idx_in_J=len(j_basis[J_val]),
                        )
                        j_basis[J_val].append(state)
                        J_val += 1.0
                    L_val += 1.0
                S_val += 1.0

    # Sort each J block to match Fortran convention: (-S_total, -L_total)
    for J_val in j_basis:
        states = j_basis[J_val]
        states.sort(key=lambda s: (-s.S_total, -s.L_total))
        for i, s in enumerate(states):
            s.idx_in_J = i

    return j_basis


# ─────────────────────────────────────────────────────────────
# MULTIPOLE transition matrix elements (electric dipole)
# ─────────────────────────────────────────────────────────────

def compute_multipole_blocks(
    l_gs: int,
    n_gs: int,
    l_core: int,
    n_core_gs: int,
    gs_terms: List[LSTerm],
    gs_parents: List[LSTerm],
    gs_cfp: np.ndarray,
) -> Dict[Tuple[float, float], np.ndarray]:
    """
    Compute MULTIPOLE (electric dipole) transition RME blocks.

    Computes transition matrix elements between:
      Ground: l_core^n_core_gs × l_gs^n_gs  (with l_core shell full)
      Excited: l_core^(n_core_gs-1) × l_gs^(n_gs+1)  (core hole)

    For L-edge XAS: l_core=1 (p), l_gs=2 (d), so ground is p^6 d^n,
    excited is p^5 d^(n+1).

    The transition is rank R=1 (electric dipole), odd parity.

    Parameters
    ----------
    l_gs : int
        Orbital angular momentum of the valence shell (2 for d).
    n_gs : int
        Number of electrons in valence shell in ground state.
    l_core : int
        Orbital angular momentum of the core shell (1 for p).
    n_core_gs : int
        Number of core electrons in ground state (6 for p^6).
    gs_terms : list of LSTerm
        LS terms of l_gs^n_gs.
    gs_parents : list of LSTerm
        LS terms of l_gs^(n_gs-1) — parents for CFP.
    gs_cfp : np.ndarray
        CFP matrix for l_gs^n_gs → l_gs^(n_gs-1).

    Returns
    -------
    dict mapping (J_bra, J_ket) → np.ndarray
        Transition matrix blocks. Bra = ground, Ket = excited.
    """
    from multitorch.angular.cfp import get_cfp_block

    R = 1  # electric dipole rank

    # ── Excited state shell terms ──
    # Core shell after excitation: l_core^(n_core_gs - 1)
    n_core_ex = n_core_gs - 1
    core_ex_block = get_cfp_block(l_core, n_core_ex)
    core_ex_terms = [LSTerm(index=t.index, S=t.S, L=t.L,
                            seniority=t.seniority,
                            label=f"{int(2*t.S+1)}{t.L_label}")
                     for t in core_ex_block.terms]

    # Valence shell after excitation: l_gs^(n_gs + 1)
    n_val_ex = n_gs + 1
    val_ex_block = get_cfp_block(l_gs, n_val_ex)
    val_ex_terms = [LSTerm(index=t.index, S=t.S, L=t.L,
                           seniority=t.seniority,
                           label=f"{int(2*t.S+1)}{t.L_label}")
                    for t in val_ex_block.terms]

    # CFP for excited valence shell: l_gs^(n_gs+1) → l_gs^n_gs (parents = gs_terms)
    val_ex_cfp = val_ex_block.cfp  # shape: (n_val_ex_terms, n_gs_terms)

    # CFP for core shell: l_core^n_core_gs → l_core^(n_core_gs-1)
    core_gs_block = get_cfp_block(l_core, n_core_gs)
    core_gs_cfp = core_gs_block.cfp  # shape: (n_core_gs_terms, n_core_ex_terms)

    # ── Build J-bases ──
    # Ground: single-shell (l_gs^n_gs, core is closed)
    S_max_gs = max(t.S for t in gs_terms)
    L_max_gs = max(t.L for t in gs_terms)
    J_min_gs = 0.0 if (2 * S_max_gs) % 2 == 0 else 0.5
    J_max_gs = S_max_gs + L_max_gs
    gs_j_basis = build_j_basis(gs_terms, J_min_gs, J_max_gs)

    # Excited: two-shell (core × valence)
    ex_j_basis = build_two_shell_j_basis(core_ex_terms, val_ex_terms)

    # ── Occupation numbers for TC0 ──
    n_annihil = n_core_gs  # electrons in annihilation shell (core in ground)
    n_create = n_val_ex    # electrons in creation shell (valence in excited)
    TC0_base = math.sqrt(n_annihil * n_create)

    # Phase: electrons between the two shells
    # IRHO=1 (core, annihilation), IRHOP=2 (valence, creation)
    # I=min=1, J=max=2. No intermediate shells → M=0.
    # N = n_create if IRHO < IRHOP, else n_annihil
    N_phase = n_create  # IRHO=1 < IRHOP=2
    if (0 + N_phase) % 2 == 0:
        TC0_base = -TC0_base

    # ── Compute transition blocks ──
    blocks: Dict[Tuple[float, float], np.ndarray] = {}

    for J_gs, gs_states in gs_j_basis.items():
        for J_ex, ex_states in ex_j_basis.items():
            # Triangle rule: |J_gs - J_ex| ≤ R=1 ≤ J_gs + J_ex
            if R > J_gs + J_ex + 1e-10 or R < abs(J_gs - J_ex) - 1e-10:
                continue

            n_bra = len(gs_states)
            n_ket = len(ex_states)
            mat = np.zeros((n_bra, n_ket), dtype=np.float64)

            for ib, sb in enumerate(gs_states):
                t_gs = sb.ls_term
                # Ground state quantum numbers
                # Shell 1 (core): closed → S_core_gs=0, L_core_gs=0
                # Shell 2 (valence): S_val_gs=t_gs.S, L_val_gs=t_gs.L
                # Total: S_total_gs=t_gs.S, L_total_gs=t_gs.L
                L_total_gs = t_gs.L
                S_total_gs = t_gs.S

                for ik, sk in enumerate(ex_states):
                    # Excited state quantum numbers
                    S_core_ex = sk.S1  # core shell spin
                    L_core_ex = sk.L1  # core shell orbital AM
                    S_val_ex = sk.S2   # valence shell spin
                    L_val_ex = sk.L2   # valence shell orbital AM
                    S_total_ex = sk.S_total
                    L_total_ex = sk.L_total

                    # Selection rules (from Fortran MUPOLE lines 1037-1053)
                    # M=1 (core shell, I=1): |L_core_gs - L_core_ex| ≤ l_core
                    # Ground core is closed: L_core_gs=0, so |L_core_ex| ≤ l_core
                    # Always true since L_core_ex ≤ l_core for l^(n-1)

                    # M=2 (valence shell, J=2): triangle on L_total with R=1
                    if abs(L_total_gs - L_total_ex) > R + 1e-10:
                        continue
                    if L_total_gs + L_total_ex < R - 1e-10:
                        continue

                    # Same S_total (M=J=2, line 1050)
                    if abs(S_total_gs - S_total_ex) > 1e-10:
                        continue

                    # ── Compute matrix element ──
                    TC = TC0_base

                    # UNCPLA for total angular momentum recoupling
                    # TC *= UNCPLA(L_total_bra, S_total_bra, J_bra, R, L_total_ket, J_ket)
                    TC *= uncpla(L_total_gs, S_total_gs, J_gs, R,
                                 L_total_ex, J_ex)

                    if abs(TC) < 1e-15:
                        continue

                    # CFP for core shell: cfp(l_core^n_core_gs ¹S | l_core^(n_core_gs-1) term)
                    # The core ground state is closed shell (only 1 term, index 0)
                    # and the core excited term is sk.term1_idx
                    if core_gs_cfp is not None and core_gs_cfp.size > 0:
                        cfp_core = core_gs_cfp[0, sk.term1_idx]
                    else:
                        cfp_core = 1.0
                    TC *= cfp_core

                    # CFP for valence shell: cfp(l_gs^(n_gs+1) ex_term | l_gs^n_gs gs_term)
                    # ITRANS=1 → use CFP2(N2, N1) = cfp(val_ex_term | gs_parent)
                    if val_ex_cfp is not None and val_ex_cfp.size > 0:
                        cfp_val = val_ex_cfp[sk.term2_idx, t_gs.index]
                    else:
                        cfp_val = 1.0
                    TC *= cfp_val

                    if abs(TC) < 1e-15:
                        continue

                    # Spin recoupling: RECPJP for spin part
                    # RECPJP(S_core_ket, 0.5, S_core_bra, S_val_bra, S_total_bra, S_val_ket)
                    # S_core_bra=0 (closed shell), S_val_bra=t_gs.S, S_total_bra=t_gs.S
                    S_core_bra = 0.0
                    S_val_bra = t_gs.S
                    if abs(S_val_bra) > 1e-10:
                        TC *= recpjp(S_core_ex, 0.5, S_core_bra,
                                     S_val_bra, S_total_gs, S_val_ex)

                    if abs(TC) < 1e-15:
                        continue

                    # Orbital recoupling: 9j symbol
                    # L_core_bra=0 (closed), L_val_bra=t_gs.L
                    L_core_bra = 0.0
                    L_val_bra = t_gs.L
                    fl_core = float(l_core)
                    fl_val = float(l_gs)

                    if abs(L_val_bra) > 1e-10:
                        # Phase: (-1)^{l_val + L_val_bra - L_val_ket}
                        phase_exp = int(round(fl_val + L_val_bra - L_val_ex))
                        if phase_exp % 2 != 0:
                            TC = -TC

                        # Factor: √((2L_core_bra+1)(2L_val_ket+1)(2L_total_bra+1)(2L_total_ket+1))
                        TC *= math.sqrt(
                            (2.0 * L_core_bra + 1.0) * (2.0 * L_val_ex + 1.0)
                            * (2.0 * L_total_gs + 1.0) * (2.0 * L_total_ex + 1.0)
                        )

                        # 9j symbol
                        TC *= wigner9j(
                            L_core_bra, L_core_ex, fl_core,
                            L_val_bra, L_val_ex, fl_val,
                            L_total_gs, L_total_ex, R,
                        )
                    else:
                        # L_val_bra = 0: use UNCPLB instead of 9j
                        # (Fortran label 560, line 1150-1152)
                        if L_core_ex > 1e-10:
                            TC *= uncplb(L_core_ex, fl_core, L_total_gs,
                                         R, fl_val, L_total_ex)
                        # else: J=1, just TC as is

                    # Phase convention for two-shell basis (Fortran compatibility)
                    phase = (-1) ** int(round(S_total_ex + L_total_ex + 1))
                    mat[ib, ik] = TC * phase

            if np.any(np.abs(mat) > 1e-15):
                blocks[(J_gs, J_ex)] = mat

    return blocks


# ─────────────────────────────────────────────────────────────
# Validation utility
# ─────────────────────────────────────────────────────────────

def validate_rme_against_reference(
    computed_rme: Dict[Tuple, torch.Tensor],
    reference_path: str,
    atol: float = 1e-6,
) -> Dict[str, bool]:
    """
    Compare computed RME values against a reference .rme_rcg file.

    For each block present in both computed and reference data, checks
    that the matrix elements agree within `atol`.

    Parameters
    ----------
    computed_rme : dict
        Dict mapping (bra_sym, op_sym, ket_sym) → torch.Tensor matrix.
    reference_path : str
        Path to reference .rme_rcg file.
    atol : float
        Absolute tolerance for agreement.

    Returns
    -------
    dict mapping block label → bool (True if agreement within atol).
    """
    from multitorch.io.read_rme import read_rme_rcg
    from pathlib import Path

    ref = read_rme_rcg(Path(reference_path))
    results = {}
    for cfg in ref.configs:
        for block in cfg.blocks:
            key = (block.bra_sym, block.op_sym, block.ket_sym)
            label = f"{block.bra_sym}/{block.op_sym}/{block.ket_sym}/{block.operator}"
            if key in computed_rme:
                diff = (computed_rme[key] - block.matrix).abs().max().item()
                results[label] = diff <= atol
            else:
                results[label] = None  # not computed
    return results
