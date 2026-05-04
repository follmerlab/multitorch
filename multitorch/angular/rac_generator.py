"""
Fortran-free RAC generator for O(3)->Oh symmetry.

Generates RACFileFull + COWAN store from scratch using:
  - CFP tables + Wigner 6j -> SHELL blocks (compute_all_shell_blocks)
  - SPIN/ORBIT blocks (compute_spin_blocks, compute_orbit_blocks)
  - O(3)->Oh ADD coupling coefficients (oh_coupling_coefficients)

This replaces the Fortran ttrac.c / ttrcg.c pipeline for the angular
structure, while being architecturally compatible with the existing
assembler.

The COWAN store layout produced here does NOT match the Fortran layout
matrix-for-matrix (the Fortran ordering is an artifact of its evaluation
order). Instead, we use a clean layout:

  Section 2 (ground manifold, gerade):
    [HAMILTONIAN blocks for each J] [10DQ blocks for each (Jb,Jk) pair]

  Section 3 (ground manifold, ungerade):
    [same structure, for odd-parity irreps]

The ADD entries reference the correct matrix_idx values for this layout.
When assembled, the result matches the Fortran output to machine precision.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from multitorch._constants import DTYPE, RY_TO_EV_FLOAT
from multitorch.angular.cfp import get_cfp_block
from multitorch.angular.rme import (
    LSTerm,
    build_j_basis,
    build_two_shell_j_basis,
    compute_all_shell_blocks,
    compute_multipole_blocks,
    compute_spin_blocks,
    compute_two_shell_exchange,
    compute_two_shell_shell_blocks,
    compute_two_shell_soc,
)
from multitorch.angular.point_group import (
    OH_IRREP_DIM,
    OH_DOUBLE_IRREP_DIM,
    OH_IRREP_DIM_ALL,
    BUTLER_LABEL,
    butler_label,
    oh_branching,
    oh_coupling_coefficients,
    oh_coupling_coefficients_full,
    oh_transition_coupling,
    _build_coupling_operator,
    _c2r_unitary,
    _real_subduction_matrix,
)
from multitorch.angular.symmetry import (
    D4H_BRANCHES_BY_OPERATOR,
    D4H_IRREP_DIM,
    D4H_TO_BUTLER,
    d4h_basis_layout,
    oh_to_d4h_subduction_matrix,
)
from multitorch.io.read_rme import (
    ADDEntry,
    IrrepInfo,
    RACBlockFull,
    RACFileFull,
)


def _get_terms(l: int, n: int) -> List[LSTerm]:
    """Load LS terms for l^n from CFP tables."""
    block = get_cfp_block(l, n)
    if block is None:
        raise ValueError(f"No CFP data for l={l}, n={n}")
    return [
        LSTerm(
            index=t.index, S=t.S, L=t.L,
            seniority=t.seniority,
            label=f"{int(2*t.S+1)}{t.L_label}",
        )
        for t in block.terms
    ]


def _j_sector_sizes(
    terms: List[LSTerm],
) -> Dict[float, int]:
    """Return dict mapping J -> number of LS terms in that J sector."""
    S_max = max(t.S for t in terms)
    L_max = max(t.L for t in terms)
    J_min = 0.0 if (2 * S_max) % 2 == 0 else 0.5
    J_max = S_max + L_max
    j_basis = build_j_basis(terms, J_min, J_max)
    return {J: len(states) for J, states in j_basis.items()}


def _irrep_block_size(
    j_sizes: Dict[float, int],
    irrep: str,
) -> int:
    """Total number of states in one Oh irrep block.

    Each J sector contributes branching(J, irrep) * n_states_at_J states.
    For the single-copy case (mult=1 for d-shell), this is just
    sum over J of n_states_at_J where irrep appears in branching(J).
    """
    total = 0
    for J, n_states in j_sizes.items():
        J_int = int(round(J))
        mult = oh_branching(J_int).get(irrep, 0)
        total += mult * n_states
    return total


def _build_hamiltonian_cowan_matrices(
    l: int,
    n: int,
    terms: List[LSTerm],
    j_sizes: Dict[float, int],
    raw_slater_ry: Dict[str, float],
    raw_zeta_ry: float,
    ry_to_ev: float,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """Build pre-assembled Hamiltonian matrices for each J sector.

    Returns (h_blocks, v11_blocks) where:
      h_blocks[J] = sum_k F^k_eV * SHELL_k(J,J) + zeta_eV * V11(J,J)
      v11_blocks[J] = V11 block for autograd decomposition

    These are the COWAN store matrices that the HAMILTONIAN ADD entries
    reference.
    """
    # Compute all SHELL blocks (k=0,2,4 for d-shell)
    all_shell = compute_all_shell_blocks(l, n)

    # Compute SPIN blocks (for SOC: V(11) = SPIN * ORBIT product)
    spin_blocks = compute_spin_blocks(terms)

    h_blocks: Dict[float, np.ndarray] = {}
    v11_blocks: Dict[float, np.ndarray] = {}

    for J, n_states in j_sizes.items():
        H = np.zeros((n_states, n_states), dtype=np.float64)

        # Coulomb: sum_k F^k * SHELL_k(J,J)
        for k in range(0, 2 * l + 1, 2):
            shell_key = (k, J, J)
            if shell_key in all_shell:
                shell_mat = all_shell[shell_key]
                # Slater integral in eV
                fk_name = f"F{k}"
                fk_ry = raw_slater_ry.get(fk_name, 0.0)
                fk_ev = fk_ry * ry_to_ev
                H += fk_ev * shell_mat

        # SOC contribution = zeta * V(11)
        # V(11) is the spin-orbit coupling matrix in the J-basis
        # V11(J,J) encodes the diagonal-in-J part of the SOC
        v11 = np.zeros((n_states, n_states), dtype=np.float64)
        spin_key = (J, J)
        if spin_key in spin_blocks:
            v11 = spin_blocks[spin_key]

        zeta_ev = raw_zeta_ry * ry_to_ev
        H += zeta_ev * v11

        h_blocks[J] = H
        v11_blocks[J] = v11

    return h_blocks, v11_blocks


def _build_cf_cowan_matrices(
    l: int,
    n: int,
    k: int,
    j_sizes: Dict[float, int],
) -> Dict[Tuple[float, float], np.ndarray]:
    """Build crystal-field SHELL_k blocks for each (J_bra, J_ket) pair.

    These are the COWAN store matrices that the 10DQ ADD entries reference.
    """
    all_shell = compute_all_shell_blocks(l, n)
    cf_blocks: Dict[Tuple[float, float], np.ndarray] = {}

    for Jb in j_sizes:
        for Jk in j_sizes:
            key = (k, Jb, Jk)
            if key in all_shell:
                cf_blocks[(Jb, Jk)] = all_shell[key]

    return cf_blocks


def _get_excited_j_sizes(
    l_val: int,
    n_val_gs: int,
    l_core: int = 1,
    n_core_gs: int = 6,
) -> Dict[float, int]:
    """Return dict mapping J -> number of states in the two-shell excited basis.

    Excited configuration: l_core^(n_core_gs-1) × l_val^(n_val_gs+1).
    """
    core_terms = _get_terms(l_core, n_core_gs - 1)
    val_terms = _get_terms(l_val, n_val_gs + 1)
    two_shell_basis = build_two_shell_j_basis(core_terms, val_terms)
    return {J: len(states) for J, states in two_shell_basis.items()}


def _build_excited_hamiltonian_cowan(
    l_val: int,
    n_val_gs: int,
    l_core: int,
    n_core_gs: int,
    raw_slater_ry: Dict[str, float],
    raw_zeta_ry: Dict[str, float],
    ry_to_ev: float,
) -> Dict[float, np.ndarray]:
    """Build pre-assembled excited-state Hamiltonian matrices for each J sector.

    The excited-state Hamiltonian includes:
      - Valence d-d Coulomb: F^k(dd) × SHELL2_k  (k=2,4)
      - Valence SOC: ζ_d × SOC_d (shell_idx=2)
      - Core SOC: ζ_p × SOC_p (shell_idx=1)
      - Inter-shell exchange: G^k(pd) × EXCHANGE_k  (k=1,3)
      - Inter-shell direct: F^k(pd) × DIRECT_k  (k=2)
      - Core-core Coulomb: F^k(pp) × SHELL1_k (not needed for p^5, single term)

    Parameters
    ----------
    raw_slater_ry : dict
        Slater integrals in Rydberg: F2_dd, F4_dd, G1_pd, G3_pd, F2_pd.
    raw_zeta_ry : dict
        SOC constants in Rydberg: {'d': zeta_d, 'p': zeta_p}.
    """
    n_core_ex = n_core_gs - 1
    n_val_ex = n_val_gs + 1

    j_sizes = _get_excited_j_sizes(l_val, n_val_gs, l_core, n_core_gs)

    h_blocks: Dict[float, np.ndarray] = {}
    for J, n_states in j_sizes.items():
        h_blocks[J] = np.zeros((n_states, n_states), dtype=np.float64)

    # Valence d-d Coulomb (k=2,4)
    for k in [2, 4]:
        fk_name = f"F{k}_dd"
        fk_ry = raw_slater_ry.get(fk_name, 0.0)
        if abs(fk_ry) < 1e-15:
            continue
        fk_ev = fk_ry * ry_to_ev
        shell_blocks = compute_two_shell_shell_blocks(
            l_val, n_val_ex, l_core, n_core_ex, k)
        for (Jb, Jk), mat in shell_blocks.items():
            if abs(Jb - Jk) < 1e-10:  # diagonal in J for Hamiltonian
                h_blocks[Jb] += fk_ev * mat

    # Valence SOC (shell_idx=2 = valence)
    zeta_d_ev = raw_zeta_ry.get('d', 0.0) * ry_to_ev
    if abs(zeta_d_ev) > 1e-15:
        soc_d = compute_two_shell_soc(
            l_val, n_val_ex, l_core, n_core_ex, shell_idx=2)
        for J, mat in soc_d.items():
            h_blocks[J] += zeta_d_ev * mat

    # Core SOC (shell_idx=1 = core)
    zeta_p_ev = raw_zeta_ry.get('p', 0.0) * ry_to_ev
    if abs(zeta_p_ev) > 1e-15:
        soc_p = compute_two_shell_soc(
            l_val, n_val_ex, l_core, n_core_ex, shell_idx=1)
        for J, mat in soc_p.items():
            h_blocks[J] += zeta_p_ev * mat

    # Inter-shell exchange G^k(pd) (k=1,3)
    for k in [1, 3]:
        gk_name = f"G{k}_pd"
        gk_ry = raw_slater_ry.get(gk_name, 0.0)
        if abs(gk_ry) < 1e-15:
            continue
        gk_ev = gk_ry * ry_to_ev
        exchange = compute_two_shell_exchange(
            l_val, n_val_ex, l_core, n_core_ex, k)
        for J, mat in exchange.items():
            h_blocks[J] += gk_ev * mat

    # Inter-shell direct Coulomb F^k(pd) (k=2 for p-d)
    # This uses the same SHELL operator but for the core shell (shell_idx=1)
    # For p^5 (single term), the core SHELL matrices are diagonal scalars,
    # so F2_pd contributes only a J-independent shift. We handle it via
    # the SHELL1 blocks (core shell U^(k) in two-shell basis).
    f2_pd_ry = raw_slater_ry.get('F2_pd', 0.0)
    if abs(f2_pd_ry) > 1e-15:
        f2_pd_ev = f2_pd_ry * ry_to_ev
        # Direct Coulomb: uses core shell U^(k=2) in the two-shell basis
        # compute_two_shell_shell_blocks with shell indices swapped:
        # we need the SHELL operator acting on the CORE shell
        # For p^5 d^(N+1), the core U^(2) is a scalar (single LS term),
        # so this is simply f2_pd * identity... but we need the proper
        # angular structure. For now, compute via the exchange infrastructure
        # which already handles the full 9j coupling.
        # Actually F^k(pd) direct uses U^(k) on both shells:
        # ⟨α|F_k|α'⟩ = n1*n2 * u^(k)(shell1) * u^(k)(shell2) * 3j^2
        # This is NOT the same as the exchange. We need a separate function.
        # F2_pd direct Coulomb is not yet implemented. Warn since it can
        # shift absolute energies by a few tenths of eV for heavy elements.
        import warnings
        warnings.warn(
            f"F2_pd = {f2_pd_ry:.4f} Ry is nonzero but the direct "
            f"Coulomb contribution is not implemented; this produces a "
            f"small systematic error in excited-state absolute energies.",
            stacklevel=2,
        )

    return h_blocks


def _build_excited_cf_cowan(
    l_val: int,
    n_val_gs: int,
    l_core: int,
    n_core_gs: int,
    cf_rank: int,
) -> Dict[Tuple[float, float], np.ndarray]:
    """Build excited-state CF SHELL_k blocks for each (J_bra, J_ket) pair.

    Crystal field acts on the valence d shell in the two-shell basis.
    """
    n_core_ex = n_core_gs - 1
    n_val_ex = n_val_gs + 1
    return compute_two_shell_shell_blocks(
        l_val, n_val_ex, l_core, n_core_ex, cf_rank)


def _build_multipole_cowan(
    l_val: int,
    n_val_gs: int,
    l_core: int,
    n_core_gs: int,
    gs_j_sizes: Dict[float, int],
    ex_j_sizes: Dict[float, int],
) -> Dict[Tuple[float, float], np.ndarray]:
    """Build MULTIPOLE transition matrices for each (J_gs, J_ex) pair.

    Wraps compute_multipole_blocks with proper term/CFP setup.
    """
    gs_terms = _get_terms(l_val, n_val_gs)
    gs_block = get_cfp_block(l_val, n_val_gs)
    gs_parents = _get_terms(l_val, n_val_gs - 1) if n_val_gs > 0 else []
    gs_cfp = gs_block.cfp if gs_block is not None else np.array([])

    return compute_multipole_blocks(
        l_val, n_val_gs, l_core, n_core_gs,
        gs_terms, gs_parents, gs_cfp)


def generate_ground_state_rac(
    l: int,
    n: int,
    raw_slater_ry: Optional[Dict[str, float]] = None,
    raw_zeta_ry: float = 0.0,
    ry_to_ev: float = RY_TO_EV_FLOAT,
    cf_rank: int = 4,
) -> Tuple[RACFileFull, List[List[torch.Tensor]]]:
    """Generate angular RME structure for single-shell d^n in Oh symmetry.

    This produces the ground-state manifold only (no excited state,
    no transition operators). The result is a self-consistent
    (RACFileFull, COWAN store) pair that the assembler can consume.

    Parameters
    ----------
    l : int
        Orbital angular momentum (2 for d-shell).
    n : int
        Number of electrons.
    raw_slater_ry : dict, optional
        Raw Slater integrals in Rydbergs: {'F0': ..., 'F2': ..., 'F4': ...}.
        If None, Hamiltonian blocks will be identity-like (debugging).
    raw_zeta_ry : float
        Raw spin-orbit coupling constant in Rydbergs.
    ry_to_ev : float
        Rydberg to eV conversion factor.
    cf_rank : int
        Crystal-field operator rank (4 for Oh cubic).

    Returns
    -------
    (rac, cowan_store)
        rac: RACFileFull with GROUND blocks for HAMILTONIAN and 10DQ.
        cowan_store: List of 4 sections (only sections 2 and 3 populated).
    """
    terms = _get_terms(l, n)
    j_sizes = _j_sector_sizes(terms)

    # Oh irreps and their Butler labels
    # For a single-shell ground state (integer J), all states are gerade (+).
    # The ungerade (-) irreps belong to the excited configuration and are
    # generated separately in Stage 1.
    # Detect half-integer J
    J_max = max(j_sizes.keys())
    is_half_int = abs(J_max - round(J_max)) > 0.1
    if is_half_int:
        oh_irreps = ['E1/2', 'E5/2', 'G3/2']
        dim_lookup = OH_DOUBLE_IRREP_DIM
    else:
        oh_irreps = ['A1', 'A2', 'E', 'T1', 'T2']
        dim_lookup = OH_IRREP_DIM
    parity = '+'

    # Compute ADD coupling coefficients for all operators (per-copy)
    add_k0 = oh_coupling_coefficients_full(J_max, 0, l=l)  # Hamiltonian (k=0)
    add_cf = oh_coupling_coefficients_full(J_max, cf_rank, l=l)  # CF (k=4)

    # Build J-sector ordering for each irrep
    # For each irrep, the states are ordered by J (ascending),
    # with all LS terms for a given J appearing as a contiguous block.
    # Each entry is (J, copy_index, n_states).
    def irrep_j_ordering(irrep):
        """Return list of (J, copy, n_states) for states in this irrep."""
        ordering = []
        for J in sorted(j_sizes.keys()):
            mult = oh_branching(J).get(irrep, 0)
            if mult > 0:
                n_states = j_sizes[J]
                for c in range(mult):
                    ordering.append((J, c, n_states))
        return ordering

    # Build COWAN store and ADD entries
    # Section 2: gerade (even parity) irreps
    # Section 3: ungerade (odd parity) irreps
    # For a single-shell ground state, odd-parity irreps appear because
    # the crystal field operator (k=4) can couple states across even J
    # values, but the Oh irreps for odd J are labeled with odd parity
    # in the Butler notation. Actually for d^n (integer J), all states
    # are gerade. But the .rme_rac has GROUND blocks for BOTH parities
    # because the assembler needs them for the excited state manifold
    # (where the core hole gives odd parity).
    #
    # For the ground-only case: section 2 holds everything.
    # We still create section 3 as empty for compatibility.

    section_2_matrices: List[torch.Tensor] = []  # gerade
    section_3_matrices: List[torch.Tensor] = []  # ungerade
    blocks: List[RACBlockFull] = []
    irrep_infos: List[IrrepInfo] = []

    # Compute Hamiltonian J-sector blocks
    if raw_slater_ry is not None:
        h_blocks, _ = _build_hamiltonian_cowan_matrices(
            l, n, terms, j_sizes, raw_slater_ry, raw_zeta_ry, ry_to_ev)
    else:
        # Default: identity blocks for each J (just for structural generation)
        h_blocks = {J: np.eye(sz) for J, sz in j_sizes.items()}

    # Compute CF SHELL blocks
    cf_blocks = _build_cf_cowan_matrices(l, n, cf_rank, j_sizes)

    # Map J-sector blocks to COWAN store indices
    # Layout: [H(J=0), H(J=1), ..., H(J_max), CF(J0,J0), CF(J0,J1), ...]

    # Only gerade section for single-shell ground state
    section_mats = section_2_matrices

    if True:  # scope block for the gerade section
        mat_idx = 1  # 1-based COWAN store index

        # Index: J -> matrix_idx for HAMILTONIAN blocks
        ham_idx: Dict[float, int] = {}
        for J in sorted(j_sizes.keys()):
            mat = h_blocks[J]
            section_mats.append(torch.as_tensor(mat, dtype=DTYPE))
            ham_idx[J] = mat_idx
            mat_idx += 1

        # Index: (Jb, Jk) -> matrix_idx for CF blocks
        cf_idx: Dict[Tuple[float, float], int] = {}
        for (Jb, Jk) in sorted(cf_blocks.keys()):
            mat = cf_blocks[(Jb, Jk)]
            section_mats.append(torch.as_tensor(mat, dtype=DTYPE))
            cf_idx[(Jb, Jk)] = mat_idx
            mat_idx += 1

        # Create RAC blocks for each irrep
        for irrep in oh_irreps:
            butler = butler_label(irrep, parity)
            j_order = irrep_j_ordering(irrep)

            if not j_order:
                continue

            block_dim = sum(n_st for _, _, n_st in j_order)

            # IRREP info
            irrep_infos.append(IrrepInfo(
                name=butler,
                kind='GROUND',
                multiplicity=block_dim,
                dim=dim_lookup[irrep],
            ))

            # HAMILTONIAN block (k=0, diagonal in J and copy)
            ham_adds = []
            bra_pos = 1  # 1-based position
            for J, copy, n_states in j_order:
                add_coeff = add_k0.get((irrep, J, copy, J, copy), 0.0)
                if abs(add_coeff) > 1e-15 and J in ham_idx:
                    ham_adds.append(ADDEntry(
                        matrix_idx=ham_idx[J],
                        bra=bra_pos,
                        ket=bra_pos,
                        nbra=n_states,
                        nket=n_states,
                        coeff=add_coeff,
                    ))
                bra_pos += n_states

            # Also add cross-copy HAMILTONIAN entries (typically zero)
            bra_pos = 1
            for ib, (Jb, cb, nb) in enumerate(j_order):
                ket_pos = 1
                for ik, (Jk, ck, nk) in enumerate(j_order):
                    if ib != ik and Jb == Jk and nb == nk:
                        # Same J, different copy: add cross-copy entry
                        add_coeff = add_k0.get(
                            (irrep, Jb, cb, Jk, ck), 0.0)
                        if Jb in ham_idx:
                            ham_adds.append(ADDEntry(
                                matrix_idx=ham_idx[Jb],
                                bra=bra_pos,
                                ket=ket_pos,
                                nbra=nb,
                                nket=nk,
                                coeff=add_coeff,
                            ))
                    ket_pos += nk
                bra_pos += nb

            if ham_adds:
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler,
                    op_sym=butler_label('A1', parity),
                    ket_sym=butler,
                    geometry='HAMILTONIAN',
                    n_bra=block_dim,
                    n_ket=block_dim,
                    add_entries=ham_adds,
                ))
            else:
                # Empty placeholder (assembler expects it)
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler,
                    op_sym=butler_label('A1', parity),
                    ket_sym=butler,
                    geometry='HAMILTONIAN',
                    n_bra=block_dim,
                    n_ket=block_dim,
                    add_entries=[],
                ))

            # 10DQ (CF) block — per-copy coefficients
            cf_adds = []
            bra_pos = 1
            for ib, (Jb, cb, n_bra) in enumerate(j_order):
                ket_pos = 1
                for ik, (Jk, ck, n_ket) in enumerate(j_order):
                    add_coeff = add_cf.get(
                        (irrep, Jb, cb, Jk, ck), 0.0)
                    if (Jb, Jk) in cf_idx:
                        cf_adds.append(ADDEntry(
                            matrix_idx=cf_idx[(Jb, Jk)],
                            bra=bra_pos,
                            ket=ket_pos,
                            nbra=n_bra,
                            nket=n_ket,
                            coeff=add_coeff,
                        ))
                    ket_pos += n_ket
                bra_pos += n_bra

            if cf_adds:
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler,
                    op_sym=butler_label('A1', parity),
                    ket_sym=butler,
                    geometry='10DQ',
                    n_bra=block_dim,
                    n_ket=block_dim,
                    add_entries=cf_adds,
                ))
            else:
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler,
                    op_sym=butler_label('A1', parity),
                    ket_sym=butler,
                    geometry='10DQ',
                    n_bra=block_dim,
                    n_ket=block_dim,
                    add_entries=[],
                ))

    # Assemble COWAN store: 4 sections (0=empty, 1=empty, 2=gerade, 3=ungerade)
    cowan_store = [
        [],  # section 0: TRANSI conf 1 (empty for ground-only)
        [],  # section 1: TRANSI conf 2 (empty for ground-only)
        section_2_matrices,
        section_3_matrices,
    ]

    rac = RACFileFull(
        irreps=irrep_infos,
        blocks=blocks,
    )

    return rac, cowan_store


# ─────────────────────────────────────────────────────────────
# D4h dispatcher helpers (Phase 1c — unified Threads 2+3)
# ─────────────────────────────────────────────────────────────
#
# These helpers replace the per-Oh-irrep block emission for `sym='d4h'`
# with a per-D4h-irrep dispatcher. The key idea: each entry in the
# `d4h_basis_layout` is a single D4h-partner basis vector (computed by
# composing `_real_subduction_matrix` with `oh_to_d4h_subduction_matrix`),
# and operator matrix elements between entries are computed by direct
# projection in real-SH basis. Cf. PORT_NOTES.md and
# `docs/D4H_DISPATCHER_PLAN.md` for the closing-of-BUG-001/-002 narrative.


def _d4h_partner_vector(
    J: float, oh_irrep_label: str, copy_idx: int,
    d4h_irrep: str, partner_idx: int,
) -> np.ndarray:
    """Build one D4h-partner basis vector (real-SH, length 2J+1).

    Composes ``_real_subduction_matrix(J, oh_irrep_label)`` (which
    splits ``D^J`` into ``Oh`` partners) with
    ``oh_to_d4h_subduction_matrix(oh_irrep_full)[d4h_irrep]`` (which
    rotates each Oh-partner triplet into D4h-partner pieces). The result
    is the basis vector keyed by
    ``(J, oh_irrep_label, copy_idx, d4h_irrep, partner_idx)``, an
    orthonormal column. Used inside ``_make_d4h_op_adds`` to project
    operators onto the per-D4h-irrep block.
    """
    parity = d4h_irrep[-1]
    oh_irrep_full = oh_irrep_label + parity
    J_int = int(round(J))
    B_oh = _real_subduction_matrix(J_int, oh_irrep_label)
    dim_oh = OH_IRREP_DIM[oh_irrep_label]
    B_copy = B_oh[:, copy_idx * dim_oh:(copy_idx + 1) * dim_oh]
    sub = oh_to_d4h_subduction_matrix(oh_irrep_full)[d4h_irrep]
    rotated = B_copy @ sub
    return np.ascontiguousarray(rotated[:, partner_idx])


# Map operator name -> (rank k, list of (Oh_route_label, butler_coeff)).
# These reproduce ``D4H_BRANCHES_BY_OPERATOR`` but in the form the
# dispatcher consumes directly: each operator's m-basis vector in
# ``D^k`` is the sum over its Oh-routes of
# ``butler_coeff × (B_oh_route @ sub_to_d4h_a1g)``.
_BUTLER_OH_TO_LABEL = {'0+': 'A1', '2+': 'E'}
_O3_TO_RANK = {'0+': 0, '2+': 2, '4+': 4}


def _d4h_op_routes(
    operator: str, target_d4h_irrep: str = 'A1g',
) -> List[Tuple[int, str, float]]:
    """Decode ``D4H_BRANCHES_BY_OPERATOR[operator]`` into (rank, oh, butler) tuples.

    Each operator in `D4H_BRANCHES_BY_OPERATOR` targets a single D4h irrep
    (currently always A1g — all CF operators are scalar in D4h). The
    `target_d4h_irrep` kwarg is plumbed through for future-proofing
    (non-A1g operators would change which D4h-Butler key is asserted)
    but in the scope of the V2 dispatcher only `'A1g'` is exercised.
    """
    target_butler = D4H_TO_BUTLER[target_d4h_irrep]
    branches = D4H_BRANCHES_BY_OPERATOR[operator]
    routes = []
    for (o3_irr, oh_butler, d4h_butler), coeff in branches.items():
        if d4h_butler != target_butler:
            raise RuntimeError(
                f"{operator}: branch target {d4h_butler!r} != "
                f"{target_butler!r} (target {target_d4h_irrep}); "
                "this dispatcher assumes a single target D4h irrep "
                "per operator."
            )
        rank = _O3_TO_RANK[o3_irr]
        oh_label = _BUTLER_OH_TO_LABEL[oh_butler]
        routes.append((rank, oh_label, coeff))
    return routes


def _d4h_operator_vector_complex(
    operator: str, rank: int, target_d4h_irrep: str = 'A1g',
) -> np.ndarray:
    """Build the operator's m-basis vector in ``D^k`` (complex basis).

    Sums Butler-weighted Oh-route contributions. Each contribution is the
    real-SH-basis target-partner (``B_oh_route @ sub_to_target``)
    multiplied by the Butler coefficient. Returns the COMPLEX-basis
    vector after applying the inverse real-SH unitary.

    OPERATOR PARITY INVARIANT: CF operators are intrinsically gerade.
    The Oh-route subduction is always looked up with parity 'g'; the
    helper does NOT take a manifold-parity kwarg. The basis vectors
    used by callers (`_d4h_partner_vector`) carry their own parity via
    the d4h_irrep label they reference.
    """
    op_vec_real = np.zeros(2 * rank + 1, dtype=np.float64)
    for r_k, oh_label, coeff in _d4h_op_routes(operator, target_d4h_irrep):
        if r_k != rank:
            continue
        oh_irrep_full = oh_label + 'g'   # CF operators are gerade by construction
        B_oh = _real_subduction_matrix(rank, oh_label)
        sub = oh_to_d4h_subduction_matrix(oh_irrep_full)[target_d4h_irrep]
        route_vec = (B_oh @ sub).flatten()
        op_vec_real += coeff * route_vec
    U_k = _c2r_unitary(rank)
    return U_k.conj().T @ op_vec_real.astype(np.complex128)


def _operator_real_matrix(
    J_b: float, J_k: float, k: int, op_vec_complex: np.ndarray,
) -> np.ndarray:
    """Build the operator matrix in real-SH basis: O_real(J_b, J_k).

    ``_build_coupling_operator`` returns the m-basis representation
    in complex basis; we unitarily rotate to real basis. By the parity
    rule, the rotated matrix is real when ``J_b + J_k + k`` is even and
    purely imaginary otherwise; we extract the real-valued payload.
    """
    O_complex = _build_coupling_operator(J_b, J_k, k, op_vec_complex)
    U_b = _c2r_unitary(int(round(J_b)))
    U_k = _c2r_unitary(int(round(J_k)))
    O_t = U_b @ O_complex @ U_k.conj().T
    if (int(round(J_b)) + int(round(J_k)) + k) % 2 == 0:
        return O_t.real
    return -O_t.imag


def _make_d4h_op_adds(
    d4h_irrep: str,
    entries: List[Tuple[float, str, int, int, int]],
    operator: str,
    ham_idx_map: Dict[float, int],
    cf_idx_map: Dict[Tuple[float, float], int],
    cf_idx_map_rank2: Dict[Tuple[float, float], int],
    *,
    target_d4h_irrep: str = 'A1g',
) -> List[ADDEntry]:
    """Build ADD entries for one (D4h irrep, operator) pair.

    SCOPE LIMITATION: assumes the operator targets a 1D / partner-symmetric
    D4h irrep (currently always A1g — all CF operators are scalar in
    D4h). The first-line filter ``entries = [e for e in entries if e[3]
    == 0]`` drops all but partner_idx=0; this is sound by Schur's
    lemma for an A1g-target operator (matrix elements are partner-
    independent), but INVALID for non-scalar targets where partners
    couple non-trivially. Pass `target_d4h_irrep != 'A1g'` only after
    auditing this assumption.

    OPERATOR PARITY INVARIANT: CF operators are gerade by construction;
    `_d4h_operator_vector_complex` hardcodes the gerade Oh subduction.
    The MANIFOLD parity comes in via `d4h_irrep` (e.g., 'A1g' for
    GROUND, 'A1u' for EXCITE) and propagates to the partner basis
    vectors through `_d4h_partner_vector`.

    Behaviour:
      - For HAMILTONIAN, uses identity (k=0); coeff
        is `sqrt(dim_d4h / (2J_b+1))` along the entry diagonal.
        Cross-(oh, copy) HAMILTONIAN matrix elements are zero by
        partner-basis orthogonality, so same_entry is necessary AND
        sufficient.
      - For TENDQ/DT/DS, computes the operator matrix in real-SH basis
        from a Butler-weighted operator vector and projects:
        ``coeff = sqrt(dim_d4h / (2J_b+1)) × <v_b | O_real | v_k>``.
    """
    # Filter to partner_idx=0 only — see SCOPE LIMITATION.
    entries = [e for e in entries if e[3] == 0]

    dim_d4h = D4H_IRREP_DIM[d4h_irrep]
    adds: List[ADDEntry] = []

    if operator == 'HAMILTONIAN':
        bra_pos = 1
        for ib, (Jb, oh_b, cb, pb, nb) in enumerate(entries):
            ket_pos = 1
            for ik, (Jk, oh_k, ck, pk, nk) in enumerate(entries):
                # Cross-(oh, copy) couplings are zero in the partner
                # basis by orthogonality; same_entry is correct.
                same_entry = (ib == ik)
                if same_entry and Jb in ham_idx_map:
                    coeff = math.sqrt(dim_d4h / (2.0 * Jb + 1.0))
                    if abs(coeff) > 1e-15:
                        adds.append(ADDEntry(
                            matrix_idx=ham_idx_map[Jb],
                            bra=bra_pos, ket=ket_pos,
                            nbra=nb, nket=nk,
                            coeff=coeff,
                        ))
                ket_pos += nk
            bra_pos += nb
        return adds

    rank_map = {'TENDQ': 4, 'DT': 4, 'DS': 2}
    if operator not in rank_map:
        raise ValueError(
            f"_make_d4h_op_adds: unknown operator {operator!r}; "
            f"expected HAMILTONIAN/TENDQ/DT/DS"
        )
    k = rank_map[operator]
    matrix_idx_map = cf_idx_map_rank2 if k == 2 else cf_idx_map
    op_vec_complex = _d4h_operator_vector_complex(operator, k, target_d4h_irrep)

    # Cache per (J_b, J_k) and per partner vector
    O_cache: Dict[Tuple[float, float], np.ndarray] = {}
    v_cache: Dict[Tuple[float, str, int, int], np.ndarray] = {}

    def get_O(Jb: float, Jk: float) -> np.ndarray:
        key = (Jb, Jk)
        if key not in O_cache:
            O_cache[key] = _operator_real_matrix(Jb, Jk, k, op_vec_complex)
        return O_cache[key]

    def get_v(J: float, oh_label: str, copy: int, partner: int) -> np.ndarray:
        key = (J, oh_label, copy, partner)
        if key not in v_cache:
            v_cache[key] = _d4h_partner_vector(
                J, oh_label, copy, d4h_irrep, partner,
            )
        return v_cache[key]

    bra_pos = 1
    for ib, (Jb, oh_b, cb, pb, nb) in enumerate(entries):
        ket_pos = 1
        for ik, (Jk, oh_k, ck, pk, nk) in enumerate(entries):
            # Triangle selection: |J_b - J_k| ≤ k ≤ J_b + J_k
            if abs(Jb - Jk) > k or Jb + Jk < k:
                ket_pos += nk
                continue
            if (Jb, Jk) not in matrix_idx_map:
                ket_pos += nk
                continue
            v_b = get_v(Jb, oh_b, cb, pb)
            v_k = get_v(Jk, oh_k, ck, pk)
            O_real = get_O(Jb, Jk)
            me = float(v_b @ O_real @ v_k)
            if abs(me) < 1e-13:
                ket_pos += nk
                continue
            coeff = math.sqrt(dim_d4h / (2.0 * Jb + 1.0)) * me
            adds.append(ADDEntry(
                matrix_idx=matrix_idx_map[(Jb, Jk)],
                bra=bra_pos, ket=ket_pos,
                nbra=nb, nket=nk,
                coeff=coeff,
            ))
            ket_pos += nk
        bra_pos += nb
    return adds


def _make_d4h_dipole_adds(
    d4h_gs: str,
    gs_entries: List[Tuple[float, str, int, int, int]],
    d4h_ex: str,
    ex_entries: List[Tuple[float, str, int, int, int]],
    op_d4h_target: str,
    multipole_idx: Dict[Tuple[float, float], int],
    factor: float,
) -> List[ADDEntry]:
    """Build TRANSI ADD entries for one (D4h gs, D4h ex, dipole sub-irrep).

    The dipole operator T1u (rank-1, ungerade) subduces to D4h-Eu (PERP,
    dim 2) + D4h-A2u (PARA, dim 1). For a given choice of `op_d4h_target`
    (one of 'Eu', 'A2u'), this helper builds the operator vector for
    that sub-irrep's partner 0 (partners are degenerate in the spectrum
    via the BAN's standard scaling) and projects matrix elements onto
    the GROUND/EXCITE D4h-partner basis vectors.

    The `factor` argument is the conventional prefactor that scales the
    matrix element relative to the multipole_blocks reference matrix:
    PERP = √(dim_Eu / dim_T1u) = √(2/3), PARA = √(dim_A2u / dim_T1u) =
    √(1/3). It maps to the existing OLD-path PERP_FACTOR / PARA_FACTOR.

    Both `gs_entries` and `ex_entries` MUST be pre-filtered to one
    partner per (J, oh, copy) — pass `[e for e in layout[d4h_*] if
    e[3] == 0]` from the caller, mirroring `_make_d4h_op_adds`'s
    SCOPE LIMITATION.
    """
    # Build the dipole operator vector (rank-1, complex basis) for the
    # target D4h sub-irrep of T1u.
    sub = oh_to_d4h_subduction_matrix('T1u')[op_d4h_target]
    op_vec_real = np.ascontiguousarray(sub[:, 0]).astype(np.float64)
    U_1 = _c2r_unitary(1)
    op_vec_complex = U_1.conj().T @ op_vec_real.astype(np.complex128)

    O_cache: Dict[Tuple[float, float], np.ndarray] = {}
    v_gs_cache: Dict[Tuple[float, str, int], np.ndarray] = {}
    v_ex_cache: Dict[Tuple[float, str, int], np.ndarray] = {}

    def get_O(Jb: float, Jk: float) -> np.ndarray:
        key = (Jb, Jk)
        if key not in O_cache:
            O_cache[key] = _operator_real_matrix(Jb, Jk, 1, op_vec_complex)
        return O_cache[key]

    def get_v(d4h_irrep: str, J: float, oh_label: str, copy: int,
              cache: Dict[Tuple[float, str, int], np.ndarray]) -> np.ndarray:
        key = (J, oh_label, copy)
        if key not in cache:
            cache[key] = _d4h_partner_vector(J, oh_label, copy, d4h_irrep, 0)
        return cache[key]

    adds: List[ADDEntry] = []
    bra_pos = 1
    for (Jb, oh_b, cb, _, nb) in gs_entries:
        ket_pos = 1
        for (Jk, oh_k, ck, _, nk) in ex_entries:
            # Triangle selection for k=1
            if abs(Jb - Jk) > 1 or Jb + Jk < 1:
                ket_pos += nk
                continue
            if (Jb, Jk) not in multipole_idx:
                ket_pos += nk
                continue
            v_b = get_v(d4h_gs, Jb, oh_b, cb, v_gs_cache)
            v_k = get_v(d4h_ex, Jk, oh_k, ck, v_ex_cache)
            O_real = get_O(Jb, Jk)
            me = float(v_b @ O_real @ v_k)
            if abs(me) < 1e-13:
                ket_pos += nk
                continue
            adds.append(ADDEntry(
                matrix_idx=multipole_idx[(Jb, Jk)],
                bra=bra_pos, ket=ket_pos,
                nbra=nb, nket=nk,
                coeff=factor * me,
            ))
            ket_pos += nk
        bra_pos += nb
    return adds


def generate_ledge_rac(
    l_val: int,
    n_val_gs: int,
    l_core: int = 1,
    n_core_gs: int = 6,
    raw_slater_gs_ry: Optional[Dict[str, float]] = None,
    raw_zeta_gs_ry: float = 0.0,
    raw_slater_ex_ry: Optional[Dict[str, float]] = None,
    raw_zeta_ex_ry: Optional[Dict[str, float]] = None,
    ry_to_ev: float = RY_TO_EV_FLOAT,
    cf_rank: int = 4,
    sym: str = 'oh',
) -> Tuple[RACFileFull, List[List[torch.Tensor]]]:
    """Generate angular RME structure for L-edge XAS.

    Produces ground state (gerade, d^n) + excited state (ungerade,
    p^5 d^(n+1)) + MULTIPOLE transition blocks.  Single-configuration
    (no charge transfer).  Output is a self-consistent
    (RACFileFull, COWAN store) pair consumable by
    ``assemble_and_diagonalize_in_memory`` with ``nconf_gs=1``.

    COWAN store layout (all in section 0 for nconf=1):
      [MULTIPOLE(J_gs,J_ex) ...] [GS_HAM(J) ...] [GS_CF(Jb,Jk) ...]
      [EX_HAM(J) ...] [EX_CF(Jb,Jk) ...]

    Parameters
    ----------
    l_val : int
        Orbital AM of valence shell (2 for d).
    n_val_gs : int
        Number of valence electrons in ground state.
    l_core : int
        Orbital AM of core shell (1 for p).
    n_core_gs : int
        Number of core electrons in ground state (6 for p^6).
    raw_slater_gs_ry : dict, optional
        Ground-state Slater integrals in Rydberg: {'F0': ..., 'F2': ..., 'F4': ...}.
    raw_zeta_gs_ry : float
        Ground-state SOC constant in Rydberg.
    raw_slater_ex_ry : dict, optional
        Excited-state Slater integrals: {'F2_dd': ..., 'F4_dd': ...,
        'G1_pd': ..., 'G3_pd': ..., 'F2_pd': ...}.
    raw_zeta_ex_ry : dict, optional
        Excited-state SOC constants: {'d': zeta_d_ry, 'p': zeta_p_ry}.
    ry_to_ev : float
        Rydberg to eV conversion.
    cf_rank : int
        Crystal field operator rank (4 for Oh).
    sym : {'oh', 'd4h'}, default 'oh'
        Target symmetry for GROUND/EXCITE blocks.

        - 'oh': single CF operator (10Dq, rank-`cf_rank`); blocks labeled
          by Oh single-group irreps. This is the historical path; backward
          compatible.
        - 'd4h': three CF operators (TENDQ, DT, DS); GROUND/EXCITE blocks
          labeled by D4h irreps; XHAM emitted by `_build_ban_from_rac`
          carries `[1.0, tendq, dt, ds]`. **Not yet implemented at the
          inner-block level** — see plan Phase 1c. The TRANSI/MULTIPOLE
          blocks are *already* D4h-aware (PERP/PARA splitting via Eu/A2u
          factors); the gap is the GROUND/EXCITE side.

    Returns
    -------
    (rac, cowan_store)
        rac: RACFileFull with TRANSI (MULTIPOLE), GROUND, and EXCITE blocks.
        cowan_store: List of 4 sections (only section 0 populated for nconf=1).
    """
    if sym not in ('oh', 'd4h'):
        raise ValueError(
            f"Unsupported symmetry {sym!r}; supported: 'oh', 'd4h'. "
            f"For trigonal symmetries see plan Phase 5."
        )
    # D4h dispatch (Phase 1c V2): per-D4h-irrep emission via
    # `_make_d4h_op_adds` and `_make_d4h_dipole_adds`. Blocks carry
    # D4h-Butler labels matching the nid8.rme_rac fixture exactly.
    # See `docs/D4H_DISPATCHER_PLAN_V2.md`.
    # ── Term data ──
    gs_terms = _get_terms(l_val, n_val_gs)
    gs_j_sizes = _j_sector_sizes(gs_terms)

    ex_j_sizes = _get_excited_j_sizes(l_val, n_val_gs, l_core, n_core_gs)

    # Detect half-integer J for ground and excited states
    J_max_gs = max(gs_j_sizes.keys())
    J_max_ex = max(ex_j_sizes.keys())
    is_half_gs = abs(J_max_gs - round(J_max_gs)) > 0.1
    is_half_ex = abs(J_max_ex - round(J_max_ex)) > 0.1

    # V2 plan §4: gate sym='d4h' on integer J + cf_rank == 4.
    # Half-integer J would require D4h double-group tables (not yet
    # tabulated). cf_rank is fixed at 4 because the d4h dispatcher uses
    # rank-4 (TENDQ/DT) and rank-2 (DS) operators by construction.
    if sym == 'd4h':
        if is_half_gs or is_half_ex:
            raise NotImplementedError(
                "sym='d4h' currently supports only even-electron-count "
                "configurations (integer J in both ground and excited "
                "manifolds). Got half-integer J — likely an odd "
                "electron count (Fe d5, Cu d9, etc.). Use sym='oh' or "
                "wait for D4h double-group support."
            )
        if cf_rank != 4:
            raise ValueError(
                f"sym='d4h' uses fixed rank-4 (TENDQ/DT) and rank-2 (DS) "
                f"operators; cf_rank={cf_rank} is ignored on this path. "
                f"Pass cf_rank=4 (default) to silence this."
            )

    if is_half_gs:
        oh_irreps_gs = ['E1/2', 'E5/2', 'G3/2']
        dim_lookup_gs = OH_DOUBLE_IRREP_DIM
    else:
        oh_irreps_gs = ['A1', 'A2', 'E', 'T1', 'T2']
        dim_lookup_gs = OH_IRREP_DIM

    if is_half_ex:
        oh_irreps_ex = ['E1/2', 'E5/2', 'G3/2']
        dim_lookup_ex = OH_DOUBLE_IRREP_DIM
    else:
        oh_irreps_ex = ['A1', 'A2', 'E', 'T1', 'T2']
        dim_lookup_ex = OH_IRREP_DIM

    # ── Compute angular coupling coefficients ──
    add_k0_gs = oh_coupling_coefficients_full(J_max_gs, 0, l=l_val)
    add_cf_gs = oh_coupling_coefficients_full(J_max_gs, cf_rank, l=l_val)

    add_k0_ex = oh_coupling_coefficients_full(J_max_ex, 0, l=l_val)
    add_cf_ex = oh_coupling_coefficients_full(J_max_ex, cf_rank, l=l_val)

    # D4h needs a second SHELL operator at rank 2 (for the DS parameter).
    # Computed only when sym='d4h' to avoid the cost in the Oh path.
    add_cf_gs_rank2 = (oh_coupling_coefficients_full(J_max_gs, 2, l=l_val)
                       if sym == 'd4h' else None)
    add_cf_ex_rank2 = (oh_coupling_coefficients_full(J_max_ex, 2, l=l_val)
                       if sym == 'd4h' else None)

    # MULTIPOLE (dipole, k=1) transition coupling: connects ground → excited
    add_transition = oh_transition_coupling(J_max_gs, J_max_ex, k=1, l=l_val)

    # ── Build J-sector ordering for irreps ──
    def irrep_j_ordering(j_sizes, irrep):
        ordering = []
        for J in sorted(j_sizes.keys()):
            mult = oh_branching(J).get(irrep, 0)
            if mult > 0:
                n_states = j_sizes[J]
                for c in range(mult):
                    ordering.append((J, c, n_states))
        return ordering

    # ── Build COWAN matrices ──
    section_0_matrices: List[torch.Tensor] = []
    mat_idx = 1  # 1-based COWAN store index

    # --- MULTIPOLE transition matrices ---
    multipole_blocks = _build_multipole_cowan(
        l_val, n_val_gs, l_core, n_core_gs, gs_j_sizes, ex_j_sizes)

    multipole_idx: Dict[Tuple[float, float], int] = {}
    for (J_gs, J_ex) in sorted(multipole_blocks.keys()):
        mat = multipole_blocks[(J_gs, J_ex)]
        section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
        multipole_idx[(J_gs, J_ex)] = mat_idx
        mat_idx += 1

    # --- Ground-state Hamiltonian matrices ---
    if raw_slater_gs_ry is not None:
        gs_h_blocks, _ = _build_hamiltonian_cowan_matrices(
            l_val, n_val_gs, gs_terms, gs_j_sizes,
            raw_slater_gs_ry, raw_zeta_gs_ry, ry_to_ev)
    else:
        gs_h_blocks = {J: np.eye(sz) for J, sz in gs_j_sizes.items()}

    gs_ham_idx: Dict[float, int] = {}
    for J in sorted(gs_j_sizes.keys()):
        mat = gs_h_blocks[J]
        section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
        gs_ham_idx[J] = mat_idx
        mat_idx += 1

    # --- Ground-state CF matrices ---
    gs_cf_blocks = _build_cf_cowan_matrices(l_val, n_val_gs, cf_rank, gs_j_sizes)

    gs_cf_idx: Dict[Tuple[float, float], int] = {}
    for (Jb, Jk) in sorted(gs_cf_blocks.keys()):
        mat = gs_cf_blocks[(Jb, Jk)]
        section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
        gs_cf_idx[(Jb, Jk)] = mat_idx
        mat_idx += 1

    # D4h: also build rank-2 CF matrices for the DS operator.
    gs_cf_idx_rank2: Dict[Tuple[float, float], int] = {}
    if sym == 'd4h':
        gs_cf_blocks_rank2 = _build_cf_cowan_matrices(
            l_val, n_val_gs, 2, gs_j_sizes,
        )
        for (Jb, Jk) in sorted(gs_cf_blocks_rank2.keys()):
            mat = gs_cf_blocks_rank2[(Jb, Jk)]
            section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
            gs_cf_idx_rank2[(Jb, Jk)] = mat_idx
            mat_idx += 1

    # --- Excited-state Hamiltonian matrices ---
    if raw_slater_ex_ry is not None and raw_zeta_ex_ry is not None:
        ex_h_blocks = _build_excited_hamiltonian_cowan(
            l_val, n_val_gs, l_core, n_core_gs,
            raw_slater_ex_ry, raw_zeta_ex_ry, ry_to_ev)
    else:
        ex_h_blocks = {J: np.eye(sz) for J, sz in ex_j_sizes.items()}

    ex_ham_idx: Dict[float, int] = {}
    for J in sorted(ex_j_sizes.keys()):
        mat = ex_h_blocks[J]
        section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
        ex_ham_idx[J] = mat_idx
        mat_idx += 1

    # --- Excited-state CF matrices ---
    ex_cf_blocks = _build_excited_cf_cowan(
        l_val, n_val_gs, l_core, n_core_gs, cf_rank)

    ex_cf_idx: Dict[Tuple[float, float], int] = {}
    for (Jb, Jk) in sorted(ex_cf_blocks.keys()):
        mat = ex_cf_blocks[(Jb, Jk)]
        section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
        ex_cf_idx[(Jb, Jk)] = mat_idx
        mat_idx += 1

    # D4h: also build rank-2 excited CF matrices for the DS operator.
    ex_cf_idx_rank2: Dict[Tuple[float, float], int] = {}
    if sym == 'd4h':
        ex_cf_blocks_rank2 = _build_excited_cf_cowan(
            l_val, n_val_gs, l_core, n_core_gs, 2,
        )
        for (Jb, Jk) in sorted(ex_cf_blocks_rank2.keys()):
            mat = ex_cf_blocks_rank2[(Jb, Jk)]
            section_0_matrices.append(torch.as_tensor(mat, dtype=DTYPE))
            ex_cf_idx_rank2[(Jb, Jk)] = mat_idx
            mat_idx += 1

    # ── Build RAC blocks and irrep infos ──
    blocks: List[RACBlockFull] = []
    irrep_infos: List[IrrepInfo] = []

    # --- TRANSI (MULTIPOLE) blocks ---
    # Dipole operator T1u splits into D4h sub-irreps:
    #   PERP (Eu, dim=2, Butler '1-'):  ADD ∝ √(dim_Eu / dim_T1u) = √(2/3)
    #   PARA (A2u, dim=1, Butler '^0-'): ADD ∝ √(dim_A2u / dim_T1u) = √(1/3)
    # Separate TRANSI blocks are created for each geometry so the assembler
    # can match triads by actor symmetry.
    PERP_FACTOR = math.sqrt(2.0 / 3.0)  # √(dim_Eu / dim_T1u)
    PARA_FACTOR = math.sqrt(1.0 / 3.0)  # √(dim_A2u / dim_T1u)

    # V2 commit 3 — per-D4h-irrep TRANSI emission via _make_d4h_dipole_adds.
    # Operates on the D4h-Butler labels emitted by commit 2 above.
    # The OLD per-Oh-irrep TRANSI loop below is gated to `sym == 'oh'`
    # by an empty-list swap; deleted in V2 commit 4.
    if sym == 'd4h':
        layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
        layout_ex = d4h_basis_layout(ex_j_sizes, parity='u')
        for d4h_gs, gs_raw in sorted(layout_gs.items()):
            gs_filtered = [e for e in gs_raw if e[3] == 0]
            if not gs_filtered:
                continue
            n_bra = sum(e[4] for e in gs_filtered)
            gs_butler = D4H_TO_BUTLER[d4h_gs]
            for d4h_ex, ex_raw in sorted(layout_ex.items()):
                ex_filtered = [e for e in ex_raw if e[3] == 0]
                if not ex_filtered:
                    continue
                n_ket = sum(e[4] for e in ex_filtered)
                ex_butler = D4H_TO_BUTLER[d4h_ex]
                for op_d4h_target, op_butler, geometry, factor in (
                    ('Eu',  '1-',  'PERP', PERP_FACTOR),
                    ('A2u', '^0-', 'PARA', PARA_FACTOR),
                ):
                    transi_adds = _make_d4h_dipole_adds(
                        d4h_gs=d4h_gs, gs_entries=gs_filtered,
                        d4h_ex=d4h_ex, ex_entries=ex_filtered,
                        op_d4h_target=op_d4h_target,
                        multipole_idx=multipole_idx,
                        factor=factor,
                    )
                    if not transi_adds:
                        continue
                    blocks.append(RACBlockFull(
                        kind='TRANSI',
                        bra_sym=gs_butler,
                        op_sym=op_butler,
                        ket_sym=ex_butler,
                        geometry=geometry,
                        n_bra=n_bra, n_ket=n_ket,
                        add_entries=transi_adds,
                    ))

    # OLD per-Oh-irrep TRANSI loop (sym='oh' only after V2 commit 3).
    _legacy_transi_gs_irreps = oh_irreps_gs if sym == 'oh' else []
    for gs_irrep in _legacy_transi_gs_irreps:
        gs_butler = butler_label(gs_irrep, '+')
        gs_j_order = irrep_j_ordering(gs_j_sizes, gs_irrep)
        if not gs_j_order:
            continue
        gs_block_dim = sum(n_st for _, _, n_st in gs_j_order)

        for ex_irrep in oh_irreps_ex:
            ex_butler = butler_label(ex_irrep, '-')
            ex_j_order = irrep_j_ordering(ex_j_sizes, ex_irrep)
            if not ex_j_order:
                continue
            ex_block_dim = sum(n_st for _, _, n_st in ex_j_order)

            # Collect Oh-level couplings for this (gs_irrep, ex_irrep) pair
            oh_couplings: Dict[Tuple[float, int, float, int], float] = {}
            for (J_gs, c_gs, _) in gs_j_order:
                for (J_ex, c_ex, _) in ex_j_order:
                    val = add_transition.get(
                        (gs_irrep, J_gs, c_gs,
                         ex_irrep, J_ex, c_ex), 0.0)
                    if abs(val) > 1e-15:
                        oh_couplings[(J_gs, c_gs, J_ex, c_ex)] = val

            if not oh_couplings:
                continue

            # Create PERP and PARA TRANSI blocks
            for d4h_factor, op_sym, geometry in [
                (PERP_FACTOR, '1-', 'PERP'),
                (PARA_FACTOR, '^0-', 'PARA'),
            ]:
                transi_adds = []
                bra_pos = 1
                for (J_gs, c_gs, n_gs) in gs_j_order:
                    ket_pos = 1
                    for (J_ex, c_ex, n_ex) in ex_j_order:
                        oh_c = oh_couplings.get(
                            (J_gs, c_gs, J_ex, c_ex), 0.0)
                        if ((J_gs, J_ex) in multipole_idx
                                and abs(oh_c) > 1e-15):
                            transi_adds.append(ADDEntry(
                                matrix_idx=multipole_idx[(J_gs, J_ex)],
                                bra=bra_pos,
                                ket=ket_pos,
                                nbra=n_gs,
                                nket=n_ex,
                                coeff=d4h_factor * oh_c,
                            ))
                        ket_pos += n_ex
                    bra_pos += n_gs

                if transi_adds:
                    blocks.append(RACBlockFull(
                        kind='TRANSI',
                        bra_sym=gs_butler,
                        op_sym=op_sym,
                        ket_sym=ex_butler,
                        geometry=geometry,
                        n_bra=gs_block_dim,
                        n_ket=ex_block_dim,
                        add_entries=transi_adds,
                    ))

    # --- D4h dispatcher (V2) — per-D4h-irrep GROUND + EXCITE emission ---
    # See `docs/D4H_DISPATCHER_PLAN_V2.md` §4. The OLD per-Oh-irrep
    # GROUND/EXCITE loops below are gated to `sym == 'oh'` and will be
    # deleted in V2 commit 4. TRANSI is still handled by the OLD path
    # (above this block); V2 commit 3 replaces it.
    if sym == 'd4h':
        # GROUND blocks (gerade, parity='g')
        layout_gs = d4h_basis_layout(gs_j_sizes, parity='g')
        for d4h_irrep, entries in sorted(layout_gs.items()):
            if not entries:
                continue
            butler = D4H_TO_BUTLER[d4h_irrep]
            block_dim = (
                sum(e[4] for e in entries) // D4H_IRREP_DIM[d4h_irrep]
            )
            irrep_infos.append(IrrepInfo(
                name=butler, kind='GROUND',
                multiplicity=block_dim,
                dim=D4H_IRREP_DIM[d4h_irrep],
            ))
            for op_name, geometry in (
                ('HAMILTONIAN', 'HAMILTONIAN'),
                ('TENDQ',       '10DQ'),
                ('DT',          'DT'),
                ('DS',          'DS'),
            ):
                op_adds = _make_d4h_op_adds(
                    d4h_irrep, entries, op_name,
                    ham_idx_map=gs_ham_idx,
                    cf_idx_map=gs_cf_idx,
                    cf_idx_map_rank2=gs_cf_idx_rank2,
                )
                if not op_adds:
                    continue
                blocks.append(RACBlockFull(
                    kind='GROUND',
                    bra_sym=butler,
                    op_sym='0+',                # D4h-A1g (scalar CF)
                    ket_sym=butler,
                    geometry=geometry,
                    n_bra=block_dim, n_ket=block_dim,
                    add_entries=op_adds,
                ))

        # EXCITE blocks (ungerade, parity='u')
        layout_ex = d4h_basis_layout(ex_j_sizes, parity='u')
        for d4h_irrep, entries in sorted(layout_ex.items()):
            if not entries:
                continue
            butler = D4H_TO_BUTLER[d4h_irrep]
            block_dim = (
                sum(e[4] for e in entries) // D4H_IRREP_DIM[d4h_irrep]
            )
            irrep_infos.append(IrrepInfo(
                name=butler, kind='EXCITE',
                multiplicity=block_dim,
                dim=D4H_IRREP_DIM[d4h_irrep],
            ))
            for op_name, geometry in (
                ('HAMILTONIAN', 'HAMILTONIAN'),
                ('TENDQ',       '10DQ'),
                ('DT',          'DT'),
                ('DS',          'DS'),
            ):
                op_adds = _make_d4h_op_adds(
                    d4h_irrep, entries, op_name,
                    ham_idx_map=ex_ham_idx,
                    cf_idx_map=ex_cf_idx,
                    cf_idx_map_rank2=ex_cf_idx_rank2,
                )
                if not op_adds:
                    continue
                blocks.append(RACBlockFull(
                    kind='GROUND',          # assembler config-1 convention
                    bra_sym=butler,
                    op_sym='0+',
                    ket_sym=butler,
                    geometry=geometry,
                    n_bra=block_dim, n_ket=block_dim,
                    add_entries=op_adds,
                ))

    # --- GROUND blocks (gerade, d^n) [OLD per-Oh-irrep path; sym='oh' only] ---
    # The d4h GROUND emission lives in the new dispatcher above; iterating
    # an empty list for d4h cleanly disables the OLD path without changing
    # the loop body's indentation. Deleted in V2 commit 4.
    _legacy_gs_irreps = oh_irreps_gs if sym == 'oh' else []
    for irrep in _legacy_gs_irreps:
        gs_butler = butler_label(irrep, '+')
        gs_j_order = irrep_j_ordering(gs_j_sizes, irrep)
        if not gs_j_order:
            continue
        block_dim = sum(n_st for _, _, n_st in gs_j_order)

        irrep_infos.append(IrrepInfo(
            name=gs_butler,
            kind='GROUND',
            multiplicity=block_dim,
            dim=dim_lookup_gs[irrep],
        ))

        # HAMILTONIAN block (k=0)
        ham_adds = _make_operator_adds(
            gs_j_order, add_k0_gs, gs_ham_idx, irrep, diagonal_only=False)
        blocks.append(RACBlockFull(
            kind='GROUND',
            bra_sym=gs_butler,
            op_sym=butler_label('A1', '+'),
            ket_sym=gs_butler,
            geometry='HAMILTONIAN',
            n_bra=block_dim,
            n_ket=block_dim,
            add_entries=ham_adds,
        ))

        # 10DQ block (CF, k=cf_rank). For Oh this is the only CF operator.
        # For D4h this is renamed to TENDQ (operator slot 2 in xham).
        cf_adds = _make_cf_adds(gs_j_order, add_cf_gs, gs_cf_idx, irrep)
        blocks.append(RACBlockFull(
            kind='GROUND',
            bra_sym=gs_butler,
            op_sym=butler_label('A1', '+'),
            ket_sym=gs_butler,
            geometry='10DQ',
            n_bra=block_dim,
            n_ket=block_dim,
            add_entries=cf_adds,
        ))

        if sym == 'd4h':
            # DT block (CF, k=cf_rank, scaled by per-Oh-irrep DT Butler ratio).
            # See multitorch/angular/symmetry.py:D4H_BRANCHES_BY_OPERATOR['DT']
            # and the strength-equality note in the rac_generator docstring
            # (sqrt(54/5) = 3*sqrt(30)/5 = TENDQ-A1 Butler factor).
            dt_a1_scale = -7.0 / 2.0 * math.sqrt(30.0) / (6.0 * math.sqrt(30.0) / 10.0)
            dt_e_scale = -5.0 / 2.0 * math.sqrt(42.0) / (6.0 * math.sqrt(30.0) / 10.0)
            ds_e_scale = -math.sqrt(70.0)  # rank-2 has no shared strength prefactor
            dt_scale = {'A1': dt_a1_scale, 'E': dt_e_scale}.get(irrep, 0.0)
            if dt_scale != 0.0:
                dt_adds_raw = _make_cf_adds(gs_j_order, add_cf_gs, gs_cf_idx, irrep)
                dt_adds = [ADDEntry(
                    matrix_idx=a.matrix_idx, bra=a.bra, ket=a.ket,
                    nbra=a.nbra, nket=a.nket, coeff=a.coeff * dt_scale,
                ) for a in dt_adds_raw]
                blocks.append(RACBlockFull(
                    kind='GROUND', bra_sym=gs_butler,
                    op_sym=butler_label('A1', '+'), ket_sym=gs_butler,
                    geometry='DT', n_bra=block_dim, n_ket=block_dim,
                    add_entries=dt_adds,
                ))
            if irrep == 'E' and add_cf_gs_rank2 is not None:
                ds_adds_raw = _make_cf_adds(
                    gs_j_order, add_cf_gs_rank2, gs_cf_idx_rank2, irrep,
                )
                ds_adds = [ADDEntry(
                    matrix_idx=a.matrix_idx, bra=a.bra, ket=a.ket,
                    nbra=a.nbra, nket=a.nket, coeff=a.coeff * ds_e_scale,
                ) for a in ds_adds_raw]
                blocks.append(RACBlockFull(
                    kind='GROUND', bra_sym=gs_butler,
                    op_sym=butler_label('A1', '+'), ket_sym=gs_butler,
                    geometry='DS', n_bra=block_dim, n_ket=block_dim,
                    add_entries=ds_adds,
                ))

    # --- EXCITE blocks (ungerade, p^5 d^(n+1)) [OLD path; sym='oh' only] ---
    # The d4h EXCITE emission lives in the new dispatcher above. Same
    # gating pattern as the GROUND loop. Deleted in V2 commit 4.
    _legacy_ex_irreps = oh_irreps_ex if sym == 'oh' else []
    for irrep in _legacy_ex_irreps:
        ex_butler = butler_label(irrep, '-')
        ex_j_order = irrep_j_ordering(ex_j_sizes, irrep)
        if not ex_j_order:
            continue
        block_dim = sum(n_st for _, _, n_st in ex_j_order)

        irrep_infos.append(IrrepInfo(
            name=ex_butler,
            kind='EXCITE',
            multiplicity=block_dim,
            dim=dim_lookup_ex[irrep],
        ))

        # HAMILTONIAN block
        ham_adds = _make_operator_adds(
            ex_j_order, add_k0_ex, ex_ham_idx, irrep, diagonal_only=False)
        blocks.append(RACBlockFull(
            kind='GROUND',  # assembler uses 'GROUND' for config-1
            bra_sym=ex_butler,
            op_sym=butler_label('A1', '+'),
            ket_sym=ex_butler,
            geometry='HAMILTONIAN',
            n_bra=block_dim,
            n_ket=block_dim,
            add_entries=ham_adds,
        ))

        # 10DQ block
        cf_adds = _make_cf_adds(ex_j_order, add_cf_ex, ex_cf_idx, irrep)
        blocks.append(RACBlockFull(
            kind='GROUND',  # assembler uses 'GROUND' for config-1
            bra_sym=ex_butler,
            op_sym=butler_label('A1', '+'),
            ket_sym=ex_butler,
            geometry='10DQ',
            n_bra=block_dim,
            n_ket=block_dim,
            add_entries=cf_adds,
        ))

        if sym == 'd4h':
            # DT/DS blocks for excited state, mirroring the ground-state path.
            dt_a1_scale = -7.0 / 2.0 * math.sqrt(30.0) / (6.0 * math.sqrt(30.0) / 10.0)
            dt_e_scale = -5.0 / 2.0 * math.sqrt(42.0) / (6.0 * math.sqrt(30.0) / 10.0)
            ds_e_scale = -math.sqrt(70.0)
            dt_scale = {'A1': dt_a1_scale, 'E': dt_e_scale}.get(irrep, 0.0)
            if dt_scale != 0.0:
                dt_adds_raw = _make_cf_adds(ex_j_order, add_cf_ex, ex_cf_idx, irrep)
                dt_adds = [ADDEntry(
                    matrix_idx=a.matrix_idx, bra=a.bra, ket=a.ket,
                    nbra=a.nbra, nket=a.nket, coeff=a.coeff * dt_scale,
                ) for a in dt_adds_raw]
                blocks.append(RACBlockFull(
                    kind='GROUND', bra_sym=ex_butler,
                    op_sym=butler_label('A1', '+'), ket_sym=ex_butler,
                    geometry='DT', n_bra=block_dim, n_ket=block_dim,
                    add_entries=dt_adds,
                ))
            if irrep == 'E' and add_cf_ex_rank2 is not None:
                ds_adds_raw = _make_cf_adds(
                    ex_j_order, add_cf_ex_rank2, ex_cf_idx_rank2, irrep,
                )
                ds_adds = [ADDEntry(
                    matrix_idx=a.matrix_idx, bra=a.bra, ket=a.ket,
                    nbra=a.nbra, nket=a.nket, coeff=a.coeff * ds_e_scale,
                ) for a in ds_adds_raw]
                blocks.append(RACBlockFull(
                    kind='GROUND', bra_sym=ex_butler,
                    op_sym=butler_label('A1', '+'), ket_sym=ex_butler,
                    geometry='DS', n_bra=block_dim, n_ket=block_dim,
                    add_entries=ds_adds,
                ))

    # ── Assemble COWAN store ──
    cowan_store = [
        section_0_matrices,  # section 0: everything for nconf=1
        [],  # section 1: unused
        [],  # section 2: unused
        [],  # section 3: unused
    ]

    rac = RACFileFull(
        irreps=irrep_infos,
        blocks=blocks,
    )

    return rac, cowan_store


def _make_operator_adds(
    j_order: List[Tuple[float, int, int]],
    add_coeffs: Dict,
    matrix_idx_map: Dict[float, int],
    irrep: str,
    diagonal_only: bool = False,
) -> List[ADDEntry]:
    """Build ADD entries for a scalar operator (HAMILTONIAN) within one irrep.

    For k=0 operators: diagonal in J, may have cross-copy terms.
    """
    adds = []
    bra_pos = 1
    for ib, (Jb, cb, nb) in enumerate(j_order):
        ket_pos = 1
        for ik, (Jk, ck, nk) in enumerate(j_order):
            if Jb != Jk:
                ket_pos += nk
                continue
            if diagonal_only and ib != ik:
                ket_pos += nk
                continue
            add_coeff = add_coeffs.get(
                (irrep, Jb, cb, Jk, ck), 0.0)
            if Jb in matrix_idx_map and abs(add_coeff) > 1e-15:
                adds.append(ADDEntry(
                    matrix_idx=matrix_idx_map[Jb],
                    bra=bra_pos,
                    ket=ket_pos,
                    nbra=nb,
                    nket=nk,
                    coeff=add_coeff,
                ))
            ket_pos += nk
        bra_pos += nb
    return adds


def _make_cf_adds(
    j_order: List[Tuple[float, int, int]],
    add_coeffs: Dict,
    matrix_idx_map: Dict[Tuple[float, float], int],
    irrep: str,
) -> List[ADDEntry]:
    """Build ADD entries for a crystal-field operator (10DQ) within one irrep."""
    adds = []
    bra_pos = 1
    for ib, (Jb, cb, n_bra) in enumerate(j_order):
        ket_pos = 1
        for ik, (Jk, ck, n_ket) in enumerate(j_order):
            add_coeff = add_coeffs.get(
                (irrep, Jb, cb, Jk, ck), 0.0)
            if (Jb, Jk) in matrix_idx_map and abs(add_coeff) > 1e-15:
                adds.append(ADDEntry(
                    matrix_idx=matrix_idx_map[(Jb, Jk)],
                    bra=bra_pos,
                    ket=ket_pos,
                    nbra=n_bra,
                    nket=n_ket,
                    coeff=add_coeff,
                ))
            ket_pos += n_ket
        bra_pos += n_bra
    return adds
