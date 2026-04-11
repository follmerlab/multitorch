"""
Hamiltonian diagonalization via torch.linalg.eigh.

Wraps LAPACK dsyevd (real symmetric eigendecomposition) for use in
multiplet calculations. Always operates in float64.

Optional Lanczos solver for very large CI spaces (stub for Phase 2,
full implementation with shift-invert and selective orthogonalization
is deferred to later).
"""
from __future__ import annotations
from typing import Optional, Tuple
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


def build_H_from_rme(
    rme_blocks: list,
    params: dict[str, float],
    sym: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Build Hamiltonian blocks from RME data and physical parameters.

    This is the central assembly step for Phase 2. It multiplies each
    RME block by its physical parameter (F2dd, F4dd, zeta, 10Dq, etc.)
    and sums them to build the full Hamiltonian.

    Parameters
    ----------
    rme_blocks : list of RACBlock
        HAMIL blocks from the .rme_rac file (or .m14 HAMILTONIAN blocks).
    params : dict
        Physical parameters keyed by operator name:
          {'HAMILTONIAN': 1.0,     # Slater+SOC = sum of F2dd×f2 + F4dd×f4 + ζ×xi
           '10Dq': cf['tendq'],
           'Dt':   cf['dt'],
           'Ds':   cf['ds']}
    sym : str
        Target symmetry ('oh', 'd4h', 'c4h').
    device : str
        PyTorch device.

    Returns
    -------
    H_dict : dict[str, torch.Tensor]
        Maps symmetry label → (N, N) Hamiltonian block tensor.

    Notes
    -----
    Full implementation requires:
      1. Reading .m14 HAMILTONIAN RME blocks (F2dd, F4dd, ζ terms separately)
      2. Multiplying each by its parameter value
      3. Adding crystal field terms (10Dq, Dt, Ds from RME blocks)
      4. Assembling off-diagonal charge-transfer blocks (XMIX)
    This stub returns empty dicts; complete implementation is Phase 2.
    """
    # TODO: full Hamiltonian assembly from RME blocks
    # Placeholder — see hamiltonian assembly notes in CLAUDE.md plan
    return {}
