"""
Track C3f — Single-fixture end-to-end parity + autograd integration test.

What this validates
-------------------
1. The full in-memory pipeline (C3b → C3c → C3d → C3e → assembler)
   reproduces the file-based ``assemble_and_diagonalize`` result on the
   nid8ct charge-transfer fixture.

   - ``Eg`` (ground eigenvalues) match at ``atol=1e-12``
   - ``Ef`` (excited eigenvalues) match at ``atol=1e-12``
   - ``T`` (transition matrix) matches at ``atol=1e-12``

2. Autograd through ``slater_scale``: ``torch.autograd.grad(Eg.sum(),
   slater_scale)`` returns a finite, nonzero gradient. This is the
   **gating test** for the entire Track C autograd story — it exercises
   every link in the chain from the autograd leaf through the COWAN
   store, assembly, and eigenvalue solver.

3. Autograd through ``soc_scale``: same contract.

4. The triad count matches the file-based result.

This test uses the nid8 ``.rcn31_out`` for raw atomic parameters
(the nid8ct fixture has no ``.rcn31_out``). The d^8 NCONF=1 params
from nid8 are used; parity is algebraically exact at scale=1.0
regardless of the exact parameter values (see
``build_cowan.py`` module docstring for the proof).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
from multitorch.atomic.scaled_params import scale_atomic_params
from multitorch.hamiltonian.assemble import (
    assemble_and_diagonalize,
    assemble_and_diagonalize_in_memory,
)
from multitorch.hamiltonian.build_cowan import build_cowan_store_in_memory
from multitorch.hamiltonian.build_rac import build_rac_in_memory
from multitorch.io.read_ban import read_ban

REFDATA = Path(__file__).parent.parent / "reference_data"
NID8CT = REFDATA / "nid8ct"
NID8CT_BAN = NID8CT / "nid8ct.ban"
NID8CT_RCG = NID8CT / "nid8ct.rme_rcg"
NID8CT_RAC = NID8CT / "nid8ct.rme_rac"
NID8_RCN31 = REFDATA / "nid8" / "nid8.rcn31_out"


# ─────────────────────────────────────────────────────────────
# Module-level fixtures
# ─────────────────────────���────────────────────────────────���──

@pytest.fixture(scope="module")
def ban():
    return read_ban(NID8CT_BAN)


@pytest.fixture(scope="module")
def raw_params():
    return read_rcn31_out_params(NID8_RCN31)


@pytest.fixture(scope="module")
def from_disk():
    """File-based reference result (the oracle)."""
    return assemble_and_diagonalize(NID8CT_RCG, NID8CT_RAC, NID8CT_BAN)


@pytest.fixture(scope="module")
def in_memory_result(ban, raw_params):
    """Full in-memory pipeline at scale=1.0."""
    scaled = scale_atomic_params(raw_params, slater_scale=1.0, soc_scale=1.0)
    rac, plan = build_rac_in_memory(
        ban, source_rac_path=NID8CT_RAC, source_rcg_path=NID8CT_RCG,
    )
    cowan = build_cowan_store_in_memory(
        scaled, raw_params, plan, source_rcg_path=NID8CT_RCG,
    )
    return assemble_and_diagonalize_in_memory(cowan, rac, ban)


# ───────��───────────────────────────��─────────────────────────
# Parity: in-memory == file-based
# ───────────────────────────────────────────���─────────────────

@pytest.mark.phase4
def test_triad_count(in_memory_result, from_disk):
    assert len(in_memory_result.triads) == len(from_disk.triads)


@pytest.mark.phase4
def test_eigenvalue_parity_Eg(in_memory_result, from_disk):
    """Ground-state eigenvalues must match file-based path."""
    for tb, td in zip(in_memory_result.triads, from_disk.triads):
        assert torch.allclose(tb.Eg, td.Eg, atol=1e-12), (
            f"Triad ({tb.gs_sym}, {tb.act_sym}, {tb.fs_sym}): "
            f"Eg max diff = {(tb.Eg - td.Eg).abs().max().item():.2e}"
        )


@pytest.mark.phase4
def test_eigenvalue_parity_Ef(in_memory_result, from_disk):
    """Excited-state eigenvalues must match file-based path."""
    for tb, td in zip(in_memory_result.triads, from_disk.triads):
        assert torch.allclose(tb.Ef, td.Ef, atol=1e-12), (
            f"Triad ({tb.gs_sym}, {tb.act_sym}, {tb.fs_sym}): "
            f"Ef max diff = {(tb.Ef - td.Ef).abs().max().item():.2e}"
        )


@pytest.mark.phase4
def test_transition_matrix_parity(in_memory_result, from_disk):
    """Transition matrix T must match file-based path."""
    for tb, td in zip(in_memory_result.triads, from_disk.triads):
        assert torch.allclose(tb.T, td.T, atol=1e-12), (
            f"Triad ({tb.gs_sym}, {tb.act_sym}, {tb.fs_sym}): "
            f"T max diff = {(tb.T - td.T).abs().max().item():.2e}"
        )


@pytest.mark.phase4
def test_symmetry_labels_match(in_memory_result, from_disk):
    """Triad symmetry labels must be identical."""
    for tb, td in zip(in_memory_result.triads, from_disk.triads):
        assert tb.gs_sym == td.gs_sym
        assert tb.act_sym == td.act_sym
        assert tb.fs_sym == td.fs_sym


# ─────────────────────────────────────────────────────────────
# Autograd: the reason Track C exists
# ─���───────────────────────────────────────────────────────────

def _build_with_grad(raw_params, ban, slater_scale, soc_scale):
    """Helper: run the full in-memory pipeline with given scale tensors."""
    scaled = scale_atomic_params(
        raw_params, slater_scale=slater_scale, soc_scale=soc_scale,
    )
    rac, plan = build_rac_in_memory(
        ban, source_rac_path=NID8CT_RAC, source_rcg_path=NID8CT_RCG,
    )
    cowan = build_cowan_store_in_memory(
        scaled, raw_params, plan, source_rcg_path=NID8CT_RCG,
    )
    return assemble_and_diagonalize_in_memory(cowan, rac, ban)


@pytest.mark.phase4
def test_autograd_through_slater_scale(ban, raw_params):
    """Backward through Eg.sum() must land on slater_scale with finite nonzero grad.

    This is the gating test for the Track C autograd story. If the
    vectorized ``assemble_matrix_from_adds`` (C3-pre), the scaled
    atomic params (C3c), or the COWAN store rebuild (C3e) sever the
    gradient, this test will fail with ``grad is None`` or zero.
    """
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale, soc_scale=1.0)

    loss = result.triads[0].Eg.sum()
    grad, = torch.autograd.grad(loss, slater_scale)

    assert torch.isfinite(grad), f"slater_scale grad is not finite: {grad}"
    assert grad.abs() > 1e-6, f"slater_scale grad is too small: {grad}"


@pytest.mark.phase4
def test_autograd_through_soc_scale(ban, raw_params):
    """Backward through Eg.sum() must land on soc_scale."""
    soc_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale=1.0, soc_scale=soc_scale)

    loss = result.triads[0].Eg.sum()
    grad, = torch.autograd.grad(loss, soc_scale)

    assert torch.isfinite(grad), f"soc_scale grad is not finite: {grad}"
    assert grad.abs() > 1e-6, f"soc_scale grad is too small: {grad}"


@pytest.mark.phase4
def test_autograd_both_scales_independent(ban, raw_params):
    """Both slater_scale and soc_scale must receive independent gradients."""
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    soc_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale, soc_scale)

    loss = result.triads[0].Eg.sum()
    grad_sl, grad_soc = torch.autograd.grad(loss, [slater_scale, soc_scale])

    assert torch.isfinite(grad_sl) and grad_sl.abs() > 1e-6
    assert torch.isfinite(grad_soc) and grad_soc.abs() > 1e-6
    # The two gradients should differ (they control different physics)
    assert not torch.allclose(grad_sl, grad_soc, atol=1e-10)


@pytest.mark.phase4
def test_autograd_all_triads(ban, raw_params):
    """Every triad's Eg.sum() must produce a finite slater_scale gradient."""
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale, soc_scale=1.0)

    for i, triad in enumerate(result.triads):
        slater_scale.grad = None  # reset between triads
        loss = triad.Eg.sum()
        grad, = torch.autograd.grad(loss, slater_scale, retain_graph=True)
        assert torch.isfinite(grad), (
            f"Triad {i} ({triad.gs_sym}): slater grad not finite"
        )
        assert grad.abs() > 1e-6, (
            f"Triad {i} ({triad.gs_sym}): slater grad too small: {grad}"
        )


@pytest.mark.phase4
def test_autograd_slater_finite_difference(ban, raw_params):
    """Verify autograd gradient matches a finite-difference estimate.

    This is a stronger check than 'grad.abs() > 1e-6': it confirms
    the *magnitude* of the autograd gradient is correct by comparing
    against (f(x+h) - f(x-h)) / 2h.  Agreement within 1% rules out
    silent detach or wrong-scale bugs.
    """
    h = 1e-5

    # Autograd gradient
    slater_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale, soc_scale=1.0)
    loss = result.triads[0].Eg.sum()
    grad_auto, = torch.autograd.grad(loss, slater_scale)

    # Finite difference: f(1+h) and f(1-h)
    result_plus = _build_with_grad(raw_params, ban, 1.0 + h, soc_scale=1.0)
    loss_plus = result_plus.triads[0].Eg.sum()

    result_minus = _build_with_grad(raw_params, ban, 1.0 - h, soc_scale=1.0)
    loss_minus = result_minus.triads[0].Eg.sum()

    grad_fd = (loss_plus - loss_minus) / (2 * h)

    # Agreement within 1% (rtol) or 1e-6 (atol, for near-zero gradients)
    assert torch.allclose(
        grad_auto, grad_fd.detach(), rtol=0.01, atol=1e-6
    ), (
        f"Autograd grad {grad_auto.item():.6e} vs finite-diff {grad_fd.item():.6e} "
        f"(rel error {abs(grad_auto.item() - grad_fd.item()) / (abs(grad_fd.item()) + 1e-30):.2%})"
    )


@pytest.mark.phase4
def test_autograd_soc_finite_difference(ban, raw_params):
    """Verify autograd gradient for soc_scale matches finite-difference estimate.

    Analogous to test_autograd_slater_finite_difference but for the
    spin-orbit coupling scale factor.
    """
    h = 1e-5

    # Autograd gradient
    soc_scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    result = _build_with_grad(raw_params, ban, slater_scale=1.0, soc_scale=soc_scale)
    loss = result.triads[0].Eg.sum()
    grad_auto, = torch.autograd.grad(loss, soc_scale)

    # Finite difference: f(1+h) and f(1-h)
    result_plus = _build_with_grad(raw_params, ban, slater_scale=1.0, soc_scale=1.0 + h)
    loss_plus = result_plus.triads[0].Eg.sum()

    result_minus = _build_with_grad(raw_params, ban, slater_scale=1.0, soc_scale=1.0 - h)
    loss_minus = result_minus.triads[0].Eg.sum()

    grad_fd = (loss_plus - loss_minus) / (2 * h)

    assert torch.allclose(
        grad_auto, grad_fd.detach(), rtol=0.01, atol=1e-6
    ), (
        f"SOC autograd grad {grad_auto.item():.6e} vs finite-diff {grad_fd.item():.6e} "
        f"(rel error {abs(grad_auto.item() - grad_fd.item()) / (abs(grad_fd.item()) + 1e-30):.2%})"
    )
