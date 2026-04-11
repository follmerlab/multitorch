"""
Transition matrix element assembly.

Given:
  - RME transition blocks T_rme (from .rme_rac TRANSI blocks)
  - Ground state eigenvectors U_g (from diagonalizing H_gs)
  - Final state eigenvectors U_f (from diagonalizing H_fs)

The transition matrix element between ground state |g> and final state |f> is:
  M[g, f] = <f | T | g> = U_f^T × T_rme × U_g

where T_rme is the RME operator matrix in the state basis.

This matches the TRANSFORMED MATRIX for TRIAD output in ttban's .oba files.
"""
from __future__ import annotations
from typing import Optional
import torch

from multitorch._constants import DTYPE


def build_transition_matrix(
    T_rme: torch.Tensor,        # (n_bra_states, n_ket_states) RME block
    U_gs: torch.Tensor,         # (n_bra_states, n_gs_eigen) ground eigenvectors
    U_fs: torch.Tensor,         # (n_ket_states, n_fs_eigen) final eigenvectors
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute transition intensities by transforming RME into eigenbasis.

    T_matrix[g, f] = (U_gs^T × T_rme × U_fs)[g, f]

    For XAS: M[g, f] = |<f|T|g>|² (pre-squared intensities)

    Parameters
    ----------
    T_rme : torch.Tensor  shape (n_bra, n_ket)
        Reduced matrix element block in the symmetry-coupled state basis.
    U_gs : torch.Tensor  shape (n_bra, n_gs)
        Ground state eigenvectors (columns) from diagonalize().
    U_fs : torch.Tensor  shape (n_ket, n_fs)
        Final state eigenvectors (columns) from diagonalize().
    device : str

    Returns
    -------
    M : torch.Tensor  shape (n_gs, n_fs)
        Transition intensity matrix (squared matrix elements).
    """
    T = T_rme.to(device=device, dtype=DTYPE)
    Ug = U_gs.to(device=device, dtype=DTYPE)
    Uf = U_fs.to(device=device, dtype=DTYPE)

    # T_eigenspace = Ug^T @ T @ Uf  →  shape (n_gs, n_fs)
    T_eigen = Ug.T @ T @ Uf

    # XAS intensity = |matrix element|²
    return T_eigen ** 2


def build_transition_matrix_ct(
    T_rme_blocks: list,         # List of RACBlock (TRANSI type)
    U_gs_per_conf: list,        # List of (n_bra_i, n_gs_i) eigenvectors per config
    U_fs_per_conf: list,        # List of (n_ket_j, n_fs_j) eigenvectors per config
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute transition matrix for charge-transfer multi-configuration case.

    In CT multiplets, each configuration pair (i, j) can have transitions.
    The total transition operator sums contributions from all allowed
    (gs_sym, trans_sym, fs_sym) triads.

    Parameters
    ----------
    T_rme_blocks : list of RACBlock
        TRANSI blocks from the .rme_rac file.
    U_gs_per_conf : list
        One eigenvector tensor per ground state configuration.
    U_fs_per_conf : list
        One eigenvector tensor per final state configuration.
    device : str

    Returns
    -------
    M : torch.Tensor  shape (n_gs_total, n_fs_total)
        Combined intensity matrix.
    """
    # Sum contributions from all configuration pairs
    # Full implementation requires mapping RACBlock (bra_sym, op_sym, ket_sym) to
    # the correct configuration pair. Deferred to complete Phase 2 implementation.
    raise NotImplementedError(
        "CT transition matrix assembly requires full Phase 2 Hamiltonian infrastructure. "
        "Use build_transition_matrix() for single-configuration cases."
    )
