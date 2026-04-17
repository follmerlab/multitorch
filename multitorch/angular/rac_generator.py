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
) -> Tuple[RACFileFull, List[List[torch.Tensor]]]:
    """Generate angular RME structure for L-edge XAS in Oh symmetry.

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

    Returns
    -------
    (rac, cowan_store)
        rac: RACFileFull with TRANSI (MULTIPOLE), GROUND, and EXCITE blocks.
        cowan_store: List of 4 sections (only section 0 populated for nconf=1).
    """
    # ── Term data ──
    gs_terms = _get_terms(l_val, n_val_gs)
    gs_j_sizes = _j_sector_sizes(gs_terms)

    ex_j_sizes = _get_excited_j_sizes(l_val, n_val_gs, l_core, n_core_gs)

    # Detect half-integer J for ground and excited states
    J_max_gs = max(gs_j_sizes.keys())
    J_max_ex = max(ex_j_sizes.keys())
    is_half_gs = abs(J_max_gs - round(J_max_gs)) > 0.1
    is_half_ex = abs(J_max_ex - round(J_max_ex)) > 0.1

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

    # ── Build RAC blocks and irrep infos ──
    blocks: List[RACBlockFull] = []
    irrep_infos: List[IrrepInfo] = []

    # --- TRANSI (MULTIPOLE) blocks ---
    # Dipole operator T1u splits into D4h sub-irreps:
    #   PERP (Eu, dim=2, Butler '1-'):  ADD = √(2/3) × oh_coupling
    #   PARA (A2u, dim=1, Butler '^0-'): ADD = √(1/3) × oh_coupling
    # Separate TRANSI blocks are created for each geometry so the assembler
    # can match triads by actor symmetry.
    PERP_FACTOR = math.sqrt(2.0 / 3.0)  # √(dim_Eu / dim_T1u)
    PARA_FACTOR = math.sqrt(1.0 / 3.0)  # √(dim_A2u / dim_T1u)

    for gs_irrep in oh_irreps_gs:
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

    # --- GROUND blocks (gerade, d^n) ---
    for irrep in oh_irreps_gs:
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

        # 10DQ block (CF, k=cf_rank)
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

    # --- EXCITE blocks (ungerade, p^5 d^(n+1)) ---
    for irrep in oh_irreps_ex:
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
