"""
Torch wrappers around the numpy angular block builders in
``multitorch.angular.rme``.

Why this module exists
----------------------
The angular block builders (``compute_shell_blocks``, ``compute_spin_blocks``,
``compute_orbit_blocks``, ``compute_multipole_blocks``) return
``Dict[(J_bra, J_ket), numpy.ndarray]``. The numpy ndarray return type makes
them unsuitable as direct inputs to the autograd-traceable Phase 5 in-memory
COWAN store builder, even though the *values* are pure geometry constants
(no autograd leaves on the upstream side).

These wrappers convert each output matrix into a ``torch.Tensor`` of dtype
``DTYPE`` so that downstream code can multiply the angular block by atomic
parameter tensors (Slater integrals ``F^k`` / ``G^k``, spin-orbit ``zeta``)
that *are* autograd leaves and have ``requires_grad=True``. The product
``angular_block * parameter`` will then propagate gradient cleanly through
the chain into ``slater_scale``, ``soc_scale`` etc.

Notes
-----
- The angular numbers themselves do not flow gradient. They depend only on
  ``l``, ``n``, ``k``, CFPs, and Wigner symbols, none of which are autograd
  leaves. The wrapped tensors are produced as plain (non-leaf) torch
  tensors with ``requires_grad=False``.
- We deliberately do *not* re-implement the angular logic in torch; the
  numpy implementations are validated against the Fortran reference
  fixtures and any rewrite would risk numerical drift.
- The dtype is forced to ``DTYPE`` (``torch.float64``) to match the
  precision used everywhere else in multitorch.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from multitorch._constants import DTYPE
from multitorch.angular.rme import (
    LSTerm,
    compute_multipole_blocks,
    compute_orbit_blocks,
    compute_shell_blocks,
    compute_spin_blocks,
)


def _wrap_block_dict(
    blocks: Dict[Tuple[float, float], np.ndarray],
) -> Dict[Tuple[float, float], torch.Tensor]:
    """Convert a dict of numpy arrays into a dict of torch tensors."""
    return {
        key: torch.as_tensor(arr, dtype=DTYPE)
        for key, arr in blocks.items()
    }


def compute_shell_blocks_torch(
    l: int,
    n: int,
    k: int,
    terms: List[LSTerm],
    uk_ls: np.ndarray,
) -> Dict[Tuple[float, float], torch.Tensor]:
    """Torch wrapper for :func:`multitorch.angular.rme.compute_shell_blocks`."""
    return _wrap_block_dict(compute_shell_blocks(l, n, k, terms, uk_ls))


def compute_spin_blocks_torch(
    terms: List[LSTerm],
) -> Dict[Tuple[float, float], torch.Tensor]:
    """Torch wrapper for :func:`multitorch.angular.rme.compute_spin_blocks`."""
    return _wrap_block_dict(compute_spin_blocks(terms))


def compute_orbit_blocks_torch(
    terms: List[LSTerm],
) -> Dict[Tuple[float, float], torch.Tensor]:
    """Torch wrapper for :func:`multitorch.angular.rme.compute_orbit_blocks`."""
    return _wrap_block_dict(compute_orbit_blocks(terms))


def compute_multipole_blocks_torch(
    l_gs: int,
    n_gs: int,
    l_core: int,
    n_core_gs: int,
    gs_terms: List[LSTerm],
    gs_parents: List[LSTerm],
    gs_cfp: np.ndarray,
) -> Dict[Tuple[float, float], torch.Tensor]:
    """Torch wrapper for :func:`multitorch.angular.rme.compute_multipole_blocks`."""
    return _wrap_block_dict(
        compute_multipole_blocks(
            l_gs, n_gs, l_core, n_core_gs, gs_terms, gs_parents, gs_cfp
        )
    )
