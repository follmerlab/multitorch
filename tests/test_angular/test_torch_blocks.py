"""
Track C3a tests for the torch wrappers in
``multitorch.angular.torch_blocks``.

Three contracts per wrapper:
  1. Numerical equivalence vs the underlying numpy implementation,
     atol=1e-12. The wrapper is supposed to be a pure type conversion;
     anything tighter than 1e-12 is unphysical because of float64
     round-off through ``torch.as_tensor``.
  2. Tensor invariants: dtype is ``DTYPE`` (float64), shape matches the
     numpy original, all entries finite, the same set of (J_bra, J_ket)
     keys is present.
  3. Downstream gradient propagation. The angular numbers themselves are
     constants, but the *product* with a parameter tensor (e.g. a Slater
     ``F^k`` or spin-orbit ``zeta`` proxy) must propagate gradient. This
     is the Track C autograd contract — the wrappers exist for the sake
     of this product, not for the wrapping itself.

Canonical small case: ``d^2`` (e.g. V³⁺), 5 LS terms, the smallest
non-trivial open d-shell. ``d^8`` is also covered for the SHELL block
to make sure non-square sub-blocks are handled.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.angular.cfp import get_cfp_block
from multitorch.angular.rme import (
    LSTerm,
    compute_multipole_blocks,
    compute_orbit_blocks,
    compute_shell_blocks,
    compute_spin_blocks,
    compute_uk_ls,
)
from multitorch.angular.torch_blocks import (
    compute_multipole_blocks_torch,
    compute_orbit_blocks_torch,
    compute_shell_blocks_torch,
    compute_spin_blocks_torch,
)


def _get_terms_and_parents(l: int, n: int):
    block_n = get_cfp_block(l, n)
    block_nm1 = get_cfp_block(l, n - 1)
    terms = [
        LSTerm(
            index=t.index,
            S=t.S,
            L=t.L,
            seniority=t.seniority,
            label=f"{int(2 * t.S + 1)}{t.L_label}",
        )
        for t in block_n.terms
    ]
    parents = [
        LSTerm(
            index=t.index,
            S=t.S,
            L=t.L,
            seniority=t.seniority,
            label=f"{int(2 * t.S + 1)}{t.L_label}",
        )
        for t in block_nm1.terms
    ]
    return terms, parents, block_n.cfp


# ─────────────────────────────────────────────────────────────
# Helpers to assert dict-of-tensors invariants
# ─────────────────────────────────────────────────────────────

def _assert_dict_equiv(
    torch_blocks, numpy_blocks, *, atol=1e-12, label=""
):
    assert set(torch_blocks.keys()) == set(numpy_blocks.keys()), (
        f"{label}: key set mismatch"
    )
    for key, t_mat in torch_blocks.items():
        n_mat = numpy_blocks[key]
        assert isinstance(t_mat, torch.Tensor), (
            f"{label} {key}: not a torch.Tensor"
        )
        assert t_mat.dtype == DTYPE, (
            f"{label} {key}: dtype is {t_mat.dtype}, expected {DTYPE}"
        )
        assert tuple(t_mat.shape) == n_mat.shape, (
            f"{label} {key}: shape {tuple(t_mat.shape)} vs {n_mat.shape}"
        )
        assert torch.isfinite(t_mat).all(), f"{label} {key}: non-finite"
        assert torch.allclose(
            t_mat, torch.as_tensor(n_mat, dtype=DTYPE), atol=atol
        ), f"{label} {key}: value mismatch"


# ─────────────────────────────────────────────────────────────
# SHELL
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase3
@pytest.mark.parametrize("n", [2, 8])
@pytest.mark.parametrize("k", [0, 2, 4])
def test_shell_torch_matches_numpy(n, k):
    terms, parents, cfp = _get_terms_and_parents(2, n)
    uk_ls = compute_uk_ls(2, n, k, terms, parents, cfp)

    numpy_blocks = compute_shell_blocks(2, n, k, terms, uk_ls)
    torch_blocks = compute_shell_blocks_torch(2, n, k, terms, uk_ls)

    _assert_dict_equiv(torch_blocks, numpy_blocks, label=f"SHELL d^{n} k={k}")
    assert len(torch_blocks) > 0, f"d^{n} k={k}: no blocks emitted"


@pytest.mark.phase3
def test_shell_torch_grad_flows_through_scalar_multiplier():
    """
    The angular block itself does not flow gradient (it's a constant),
    but multiplying by an autograd-leaf scalar must produce a finite,
    correct gradient on the scalar. This is the actual Track C
    contract: angular_block * F^k must propagate grad into F^k.
    """
    terms, parents, cfp = _get_terms_and_parents(2, 8)
    uk_ls = compute_uk_ls(2, 8, 2, terms, parents, cfp)
    blocks = compute_shell_blocks_torch(2, 8, 2, terms, uk_ls)
    assert len(blocks) > 0

    fk = torch.tensor(0.85, dtype=DTYPE, requires_grad=True)
    loss = sum((fk * mat).sum() for mat in blocks.values())
    loss.backward()

    assert fk.grad is not None
    assert torch.isfinite(fk.grad)
    # d(loss)/d(fk) = sum over all blocks of sum(mat); sanity check it's
    # nonzero (the SHELL k=2 block is definitely not all zeros for d^8).
    expected = sum(
        torch.as_tensor(
            compute_shell_blocks(2, 8, 2, terms, uk_ls)[key], dtype=DTYPE
        ).sum()
        for key in blocks
    )
    assert torch.isclose(fk.grad, expected, atol=1e-12)
    assert fk.grad.abs() > 1e-6


# ─────────────────────────────────────────────────────────────
# SPIN
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase3
@pytest.mark.parametrize("n", [2, 8])
def test_spin_torch_matches_numpy(n):
    terms, _, _ = _get_terms_and_parents(2, n)
    numpy_blocks = compute_spin_blocks(terms)
    torch_blocks = compute_spin_blocks_torch(terms)
    _assert_dict_equiv(torch_blocks, numpy_blocks, label=f"SPIN d^{n}")
    assert len(torch_blocks) > 0


@pytest.mark.phase3
def test_spin_torch_grad_flows_through_zeta_proxy():
    """
    The Track C autograd story routes spin-orbit ``zeta`` as a multiplier
    onto SPIN/ORBIT blocks. Verify the wrapped tensor lets gradient flow
    into a scalar zeta proxy.
    """
    terms, _, _ = _get_terms_and_parents(2, 8)
    blocks = compute_spin_blocks_torch(terms)

    zeta = torch.tensor(0.072, dtype=DTYPE, requires_grad=True)  # ~Ni 3d ζ in Ry
    loss = sum((zeta * mat).sum() for mat in blocks.values())
    loss.backward()

    assert zeta.grad is not None
    assert torch.isfinite(zeta.grad)
    expected = sum(
        torch.as_tensor(compute_spin_blocks(terms)[key], dtype=DTYPE).sum()
        for key in blocks
    )
    assert torch.isclose(zeta.grad, expected, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# ORBIT
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase3
@pytest.mark.parametrize("n", [2, 8])
def test_orbit_torch_matches_numpy(n):
    terms, _, _ = _get_terms_and_parents(2, n)
    numpy_blocks = compute_orbit_blocks(terms)
    torch_blocks = compute_orbit_blocks_torch(terms)
    _assert_dict_equiv(torch_blocks, numpy_blocks, label=f"ORBIT d^{n}")
    assert len(torch_blocks) > 0


@pytest.mark.phase3
def test_orbit_torch_grad_flows_through_scalar_multiplier():
    terms, _, _ = _get_terms_and_parents(2, 8)
    blocks = compute_orbit_blocks_torch(terms)

    g = torch.tensor(1.25, dtype=DTYPE, requires_grad=True)
    loss = sum((g * mat).sum() for mat in blocks.values())
    loss.backward()

    assert g.grad is not None
    assert torch.isfinite(g.grad)


# ─────────────────────────────────────────────────────────────
# MULTIPOLE (electric dipole, two-shell)
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase3
def test_multipole_torch_matches_numpy_d8():
    """
    Multipole blocks for the L-edge XAS of d^8 (Ni²⁺): ground p^6 d^8,
    excited p^5 d^9.
    """
    gs_terms, gs_parents, gs_cfp = _get_terms_and_parents(2, 8)

    numpy_blocks = compute_multipole_blocks(
        l_gs=2, n_gs=8, l_core=1, n_core_gs=6,
        gs_terms=gs_terms, gs_parents=gs_parents, gs_cfp=gs_cfp,
    )
    torch_blocks = compute_multipole_blocks_torch(
        l_gs=2, n_gs=8, l_core=1, n_core_gs=6,
        gs_terms=gs_terms, gs_parents=gs_parents, gs_cfp=gs_cfp,
    )

    _assert_dict_equiv(
        torch_blocks, numpy_blocks, label="MULTIPOLE d^8 → p^5 d^9"
    )
    assert len(torch_blocks) > 0


@pytest.mark.phase3
def test_multipole_torch_grad_flows_through_radial_dipole_proxy():
    """
    The dipole radial integral R^1(2p, 3d) multiplies the multipole
    angular block in C3e. Verify gradient propagates into a scalar
    R-proxy.
    """
    gs_terms, gs_parents, gs_cfp = _get_terms_and_parents(2, 8)
    blocks = compute_multipole_blocks_torch(
        l_gs=2, n_gs=8, l_core=1, n_core_gs=6,
        gs_terms=gs_terms, gs_parents=gs_parents, gs_cfp=gs_cfp,
    )

    R = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
    loss = sum((R * mat).sum() for mat in blocks.values())
    loss.backward()

    assert R.grad is not None
    assert torch.isfinite(R.grad)
    assert R.grad.abs() > 1e-6


# ─────────────────────────────────────────────────────────────
# Cross-cutting invariant
# ─────────────────────────────────────────────────────────────

@pytest.mark.phase3
def test_wrapped_tensors_are_not_leaves_with_requires_grad():
    """
    The wrapped tensors are pure constants. They must not silently
    arrive with ``requires_grad=True`` (which would create dangling
    leaves in the autograd graph).
    """
    terms, parents, cfp = _get_terms_and_parents(2, 8)
    uk_ls = compute_uk_ls(2, 8, 2, terms, parents, cfp)
    for mat in compute_shell_blocks_torch(2, 8, 2, terms, uk_ls).values():
        assert mat.requires_grad is False
    for mat in compute_spin_blocks_torch(terms).values():
        assert mat.requires_grad is False
    for mat in compute_orbit_blocks_torch(terms).values():
        assert mat.requires_grad is False
