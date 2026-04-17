"""
Hamiltonian diagonalization via torch.linalg.eigh.

Wraps LAPACK dsyevd (real symmetric eigendecomposition) for use in
multiplet calculations. Always operates in float64.

Includes a degeneracy guard for autograd stability: the backward pass
of ``torch.linalg.eigh`` divides by (λ_i − λ_j), which produces
NaN/inf when eigenvalues are degenerate. When gradients are needed,
``safe_eigh`` lifts near-degeneracies with a tiny diagonal perturbation.
"""
from __future__ import annotations
from typing import Tuple
import torch

from multitorch._constants import DTYPE

# Perturbation magnitude for degeneracy lifting.  1e-12 eV is ~8 orders
# of magnitude below typical crystal-field splittings (~1e-4 eV) so it
# doesn't affect any physical observable, but it's large enough that
# (λ_i − λ_j) ≫ machine epsilon in the backward pass.
_DEGEN_EPS = 1e-12


def safe_eigh(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition with degeneracy guard for autograd stability.

    When ``H.requires_grad`` is True, adds a tiny diagonal perturbation
    (linearly spaced, O(1e-12)) to lift exact eigenvalue degeneracies
    before calling ``torch.linalg.eigh``. This prevents NaN/inf in the
    backward pass without measurably affecting forward values.

    When ``H.requires_grad`` is False, delegates directly to
    ``torch.linalg.eigh`` with no perturbation (preserving exact
    Fortran-matching numerics).
    """
    H = H.to(dtype=DTYPE)
    if not torch.allclose(H, H.T, atol=1e-10):
        H = 0.5 * (H + H.T)

    if H.requires_grad:
        n = H.shape[0]
        # Linearly spaced perturbation so no two diagonal elements get
        # the same shift — this guarantees distinct eigenvalues even in
        # fully degenerate subspaces.
        perturb = torch.linspace(0, _DEGEN_EPS * n, n, dtype=DTYPE, device=H.device)
        H = H + torch.diag(perturb)

    return torch.linalg.eigh(H)


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
    return safe_eigh(H)


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
