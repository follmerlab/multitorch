"""
Charge transfer Hamiltonian block assembly.

In a charge transfer multiplet calculation, multiple electronic configurations
(e.g., d8, d9L, d10L2 for LMCT) are mixed via off-diagonal interaction
matrix elements V (the charge transfer integral).

The full Hamiltonian has a block structure:
  H_full = [[H_conf1 + E1*I,   V12,           0           ]
             [V12†,            H_conf2 + E2*I, V23         ]
             [0,                V23†,           H_conf3 + E3*I]]

Where each H_conf = HAMILTONIAN × 1.0 + 10DQ × tendq + DT × dt + DS × ds

The Hamiltonian is assembled from:
  - .rme_rcg: COWAN store (indexed sparse matrices from ttrcg)
  - .rme_rac: ADD entries specifying how to combine COWAN matrices
  - .ban: XHAM/XMIX scaling, energy offsets, triads

Reference: ttban_exact.f subroutines MASTER, PAIRIN, COWAN, ETR, sp_BUILD
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from multitorch._constants import DTYPE


@dataclass
class TriadSpec:
    """One symmetry triad from the .ban file."""
    gs_sym: str     # ground state irrep (e.g., '0+')
    act_sym: str    # actor irrep (e.g., '1-')
    fs_sym: str     # final state irrep (e.g., '1-')


@dataclass
class AssembledTriad:
    """Assembled Hamiltonian and transition matrices for one triad."""
    triad: TriadSpec
    H_gs: torch.Tensor      # ground state Hamiltonian (n_gs_total, n_gs_total)
    H_fs: torch.Tensor      # final state Hamiltonian (n_fs_total, n_fs_total)
    T: torch.Tensor          # transition matrix (n_gs_total, n_fs_total)
    gs_conf_sizes: List[int]  # sizes per configuration
    fs_conf_sizes: List[int]
    gs_conf_labels: List[int]  # config index (1-based) for each state


def assemble_hamiltonian_block(
    rac_blocks,
    cowan_sections: List[List[torch.Tensor]],
    operator_names: List[str],
    xham_values: List[float],
    n_dim: int,
    energy_offset: float = 0.0,
    cowan_section_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble one diagonal Hamiltonian block for a single configuration.

    Sums: H = Σ_op xham[op] × assemble(rac_block[op], cowan_section)  +  E_offset × I

    Parameters
    ----------
    rac_blocks : list of RACBlockFull
        One block per operator (HAMILTONIAN, 10DQ, DT, DS), matched by order.
    cowan_sections : list of list of torch.Tensor
        The full COWAN store.
    operator_names : list of str
        Operator names matching rac_blocks order.
    xham_values : list of float
        XHAM scaling values (same order as operator_names).
    n_dim : int
        Matrix dimension.
    energy_offset : float
        Diagonal energy shift (EG or EF value).
    cowan_section_idx : int
        Which COWAN section these blocks reference.

    Returns
    -------
    torch.Tensor shape (n_dim, n_dim)
    """
    from multitorch.io.read_rme import assemble_matrix_from_adds

    H = torch.zeros(n_dim, n_dim, dtype=DTYPE)

    section = cowan_sections[cowan_section_idx] if cowan_section_idx < len(cowan_sections) else []

    for block, xval in zip(rac_blocks, xham_values):
        if block is not None and block.add_entries and xval != 0.0:
            contrib = assemble_matrix_from_adds(
                block.add_entries, section, n_dim, n_dim, scale=xval,
            )
            H += contrib

    # Add energy offset
    if energy_offset != 0.0:
        H += energy_offset * torch.eye(n_dim, dtype=DTYPE)

    # Symmetrize
    H = 0.5 * (H + H.T)

    return H


def assemble_mixing_block(
    rac_blocks,
    cowan_sections: List[List[torch.Tensor]],
    xmix_values: List[float],
    n_bra: int,
    n_ket: int,
    cowan_section_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble one off-diagonal mixing block between configurations.

    The mixing block is:  V = Σ_channel xmix[ch] × assemble(hybr_block[ch], cowan_section)

    Parameters
    ----------
    rac_blocks : list of RACBlockFull
        One per hybridization channel (B1HYBR, A1HYBR, B2HYBR, EHYBR).
    cowan_sections : list of list of torch.Tensor
    xmix_values : list of float
        Charge transfer integral per channel.
    n_bra, n_ket : int
        Block dimensions.
    cowan_section_idx : int
        Which COWAN section these blocks reference.

    Returns
    -------
    torch.Tensor shape (n_bra, n_ket)
    """
    from multitorch.io.read_rme import assemble_matrix_from_adds

    V = torch.zeros(n_bra, n_ket, dtype=DTYPE)
    section = cowan_sections[cowan_section_idx] if cowan_section_idx < len(cowan_sections) else []

    for block, xval in zip(rac_blocks, xmix_values):
        if block is not None and block.add_entries and xval != 0.0:
            contrib = assemble_matrix_from_adds(
                block.add_entries, section, n_bra, n_ket, scale=xval,
            )
            V += contrib

    return V


def assemble_transition_block(
    rac_block,
    cowan_sections: List[List[torch.Tensor]],
    n_bra: int,
    n_ket: int,
    cowan_section_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble one transition matrix block for a single configuration pair.

    Parameters
    ----------
    rac_block : RACBlockFull
        The TRANSI block for this (gs_sym, act_sym, fs_sym) triad.
    cowan_sections : list of list of torch.Tensor
    n_bra, n_ket : int
    cowan_section_idx : int

    Returns
    -------
    torch.Tensor shape (n_bra, n_ket)
    """
    from multitorch.io.read_rme import assemble_matrix_from_adds

    section = cowan_sections[cowan_section_idx] if cowan_section_idx < len(cowan_sections) else []

    if rac_block is None or not rac_block.add_entries:
        return torch.zeros(n_bra, n_ket, dtype=DTYPE)

    return assemble_matrix_from_adds(
        rac_block.add_entries, section, n_bra, n_ket, scale=1.0,
    )


def build_ct_energy_offsets(
    nconf: int,
    delta_lmct: Optional[float] = None,
    u_lmct: Optional[float] = None,
    delta_mlct: Optional[float] = None,
    u_mlct: Optional[float] = None,
    state: str = "ground",
) -> List[float]:
    """
    Compute energy offsets for each configuration in the CT Hamiltonian.

    Matches pyctm/write_BAN.py::writeBAN() DEF EG/EF logic:
      For LMCT:
        EG_conf = [0, delta_lmct, 2*delta_lmct - U, ...]
        EF_conf = [0, delta_lmct - U, ...]

    Parameters
    ----------
    nconf : int
        Number of configurations.
    delta_lmct : float
        Charge transfer energy δ for LMCT (eV).
    u_lmct : float
        Coulomb repulsion U for LMCT (eV).
    state : str
        'ground' or 'final'.

    Returns
    -------
    offsets : list of float
        Length nconf, energy offset for each configuration in eV.
    """
    offsets = [0.0] * nconf
    if delta_lmct is not None and u_lmct is not None:
        offsets[-1] = delta_lmct
        if state == "final":
            offsets[-1] = delta_lmct - u_lmct
    if delta_mlct is not None and u_mlct is not None:
        offsets[0] = delta_mlct
        if state == "final":
            offsets[0] = delta_mlct + u_mlct
    base = offsets[0]
    return [e - base for e in offsets]
