"""
Hamiltonian diagonalization via torch.linalg.eigh.

Wraps LAPACK dsyevd (real symmetric eigendecomposition) for use in
multiplet calculations. Always operates in float64.
"""
from __future__ import annotations
from typing import Tuple
import torch

from multitorch._constants import DTYPE


def diagonalize(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exactly diagonalize a real symmetric matrix H.

    Parameters
    ----------
    H : torch.Tensor  shape (N, N)
        Real symmetric Hamiltonian in float64.

    Returns
    -------
    eigenvalues : torch.Tensor  shape (N,)
        Eigenvalues in ascending order.
    eigenvectors : torch.Tensor  shape (N, N)
        Columns are normalized eigenvectors. eigenvectors[:, k] is the
        k-th eigenvector corresponding to eigenvalues[k].
    """
    H = H.to(dtype=DTYPE)
    if not torch.allclose(H, H.T, atol=1e-10):
        # Symmetrize in case of floating point asymmetry
        H = 0.5 * (H + H.T)
    return torch.linalg.eigh(H)


def diagonalize_block(
    H_dict: dict[str, torch.Tensor]
) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Diagonalize a collection of Hamiltonian blocks keyed by symmetry label.

    Parameters
    ----------
    H_dict : dict
        Mapping from symmetry label (e.g. '0+', '1-') to (N, N) float64
        Hamiltonian block.

    Returns
    -------
    dict mapping symmetry label → (eigenvalues, eigenvectors).
    """
    return {sym: diagonalize(H) for sym, H in H_dict.items()}


