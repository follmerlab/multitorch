"""
Track C3-pre regression and gradient tests for assemble_matrix_from_adds.

Why this file exists:
  The previous implementation of `multitorch.io.read_rme.assemble_matrix_from_adds`
  was a Python double loop that called `float(src[jr, jc])` on every
  element. This severed the autograd graph from `cowan_section` to the
  assembled matrix, making the entire Phase 5 (Track C) autograd story
  impossible — backward passes through `slater_scale`, `cf['tendq']`, etc
  would always return zero or None.

  C3-pre vectorizes the function: each ADD entry now does one slice add
  (`mat[r0:r0+nr, c0:c0+nc] += factor * src[...]`). This file locks in:
    1. Numerical equivalence vs the legacy loop on every block of the
       nid8ct.rme_rac fixture (226 blocks × every operator). Tolerance
       is exact (atol=0) — the operations are mathematically identical.
    2. Gradient propagation: build a `cowan_section` with
       `requires_grad=True`, call the new function, sum, backward, and
       assert the leaf tensor's `.grad` is finite, nonzero, and
       analytically correct. The legacy implementation returned zero
       grads here.
    3. Per-ADD correctness on a hand-crafted minimal case to catch any
       subtle off-by-one in the slice arithmetic that the fixture
       comparison might miss.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.io.read_rme import (
    ADDEntry,
    assemble_matrix_from_adds,
    read_cowan_store,
    read_rme_rac_full,
)

REFDATA = Path(__file__).parent.parent / "reference_data" / "nid8ct"


# ─────────────────────────────────────────────────────────────
# Legacy implementation kept here for regression comparison only.
# Do NOT use in production code. This is the *exact* algorithm
# that shipped before C3-pre, including the gradient-severing
# `float(src[jr, jc])` call.
# ─────────────────────────────────────────────────────────────
def _assemble_matrix_from_adds_legacy(
    add_entries, cowan_section, n_bra, n_ket, scale=1.0
):
    mat = torch.zeros(n_bra, n_ket, dtype=DTYPE)
    for add in add_entries:
        idx = add.matrix_idx - 1
        if idx < 0 or idx >= len(cowan_section):
            continue
        src = cowan_section[idx]
        r0 = add.bra - 1
        c0 = add.ket - 1
        src_rows = min(add.nbra, src.shape[0])
        src_cols = src.shape[1] if len(src.shape) > 1 else 1
        factor = add.coeff * scale
        if len(src.shape) == 1 or src.shape[1] == 0:
            continue
        for jr in range(src_rows):
            for jc in range(src_cols):
                val = float(src[jr, jc])
                if val != 0.0:
                    r = r0 + jr
                    c = c0 + jc
                    if 0 <= r < n_bra and 0 <= c < n_ket:
                        mat[r, c] += factor * val
    return mat


@pytest.fixture(scope="module")
def nid8ct_cowan_and_rac():
    cowan = read_cowan_store(REFDATA / "nid8ct.rme_rcg")
    rac = read_rme_rac_full(REFDATA / "nid8ct.rme_rac")
    return cowan, rac


@pytest.mark.phase2
def test_vectorized_matches_legacy_on_every_nid8ct_block(nid8ct_cowan_and_rac):
    """
    Vectorized assemble_matrix_from_adds must produce bit-identical
    output to the legacy double loop on every ADD-bearing block of the
    nid8ct.rme_rac fixture, across both COWAN sections that the file
    references. atol=0 — these are mathematically identical operations.
    """
    cowan, rac = nid8ct_cowan_and_rac
    assert len(rac.blocks) == 226, "fixture changed"

    blocks_compared = 0
    for block in rac.blocks:
        if not block.add_entries:
            continue
        # Try every COWAN section the ADD entries might reference. The
        # caller in assemble.py knows which section to use; we don't,
        # so iterate over both sections that contain the indices we see.
        max_idx = max(a.matrix_idx for a in block.add_entries)
        for section_idx, section in enumerate(cowan):
            if max_idx > len(section):
                continue
            new = assemble_matrix_from_adds(
                block.add_entries, section, block.n_bra, block.n_ket
            )
            old = _assemble_matrix_from_adds_legacy(
                block.add_entries, section, block.n_bra, block.n_ket
            )
            assert torch.equal(new, old), (
                f"mismatch on block {block.kind} {block.bra_sym} "
                f"{block.geometry} (section {section_idx})"
            )
            blocks_compared += 1

    # Sanity check: we actually exercised a meaningful number of blocks.
    # If this drops to 0 the test is silently passing without comparing
    # anything.
    assert blocks_compared > 100, (
        f"only {blocks_compared} blocks compared — fixture/section "
        "indexing may have changed"
    )


@pytest.mark.phase2
def test_vectorized_matches_legacy_with_scale(nid8ct_cowan_and_rac):
    """Same as above but with a non-trivial scale factor."""
    cowan, rac = nid8ct_cowan_and_rac
    scale = 0.7345

    n_compared = 0
    for block in rac.blocks[:50]:  # subset is enough — algorithm is the same
        if not block.add_entries:
            continue
        max_idx = max(a.matrix_idx for a in block.add_entries)
        for section in cowan:
            if max_idx > len(section):
                continue
            new = assemble_matrix_from_adds(
                block.add_entries, section, block.n_bra, block.n_ket, scale=scale
            )
            old = _assemble_matrix_from_adds_legacy(
                block.add_entries, section, block.n_bra, block.n_ket, scale=scale
            )
            assert torch.equal(new, old)
            n_compared += 1
            break

    assert n_compared > 10


@pytest.mark.phase2
def test_gradient_propagation_through_cowan_section():
    """
    The gating contract for Track C autograd: build a cowan_section
    whose source matrix has requires_grad=True, run
    assemble_matrix_from_adds, sum the result, backward, and assert
    that the source tensor receives a finite, nonzero, analytically
    correct gradient.

    The legacy implementation called `float(src[jr, jc])`, which
    returned None for `src.grad` (gradient severance). This test would
    fail under the legacy implementation.
    """
    src = torch.tensor(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    cowan_section = [src]
    add_entries = [
        # Place the entire 3x3 source at (0, 0) of a 5x5 matrix with coeff=2
        ADDEntry(matrix_idx=1, bra=1, ket=1, nbra=3, nket=3, coeff=2.0),
        # Place rows [0:2, 0:2] of the source at (3, 3) with coeff=-1
        ADDEntry(matrix_idx=1, bra=4, ket=4, nbra=2, nket=2, coeff=-1.0),
    ]

    mat = assemble_matrix_from_adds(add_entries, cowan_section, 5, 5)

    # Forward sanity: mat[0:3, 0:3] = 2 * src; mat[3:5, 3:5] = -1 * src[0:2, 0:2]
    expected = torch.zeros(5, 5, dtype=DTYPE)
    expected[0:3, 0:3] = 2.0 * src.detach()
    expected[3:5, 3:5] = -1.0 * src.detach()[0:2, 0:2]
    assert torch.allclose(mat, expected)

    # Backward: d(sum mat)/d(src[i,j]) =
    #   2  for every (i,j) in src
    #   plus -1 for (i,j) in src[0:2, 0:2]
    loss = mat.sum()
    loss.backward()

    grad = src.grad
    assert grad is not None, "gradient severance — assemble_matrix_from_adds is not autograd-clean"
    assert torch.isfinite(grad).all()

    expected_grad = torch.full((3, 3), 2.0, dtype=DTYPE)
    expected_grad[0:2, 0:2] += -1.0
    assert torch.allclose(grad, expected_grad), (
        f"gradient mismatch:\n  got      {grad}\n  expected {expected_grad}"
    )


@pytest.mark.phase2
def test_gradient_propagation_through_scale():
    """
    `scale` may be passed as a torch scalar tensor with requires_grad=True
    (this is one of the gradient entry points for Track C — slater_scale
    flows in here). Verify the gradient is correct.
    """
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    cowan_section = [src]
    add_entries = [
        ADDEntry(matrix_idx=1, bra=1, ket=1, nbra=2, nket=2, coeff=3.0),
    ]
    scale = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)

    mat = assemble_matrix_from_adds(add_entries, cowan_section, 2, 2, scale=scale)
    loss = mat.sum()
    loss.backward()

    # mat = 3 * 1.5 * src = 4.5 * src
    # d(sum mat)/d(scale) = 3 * (sum src) = 3 * 10 = 30
    assert scale.grad is not None
    assert torch.isclose(scale.grad, torch.tensor(30.0, dtype=DTYPE))


@pytest.mark.phase2
def test_minimal_handcrafted_slice_arithmetic():
    """
    Hand-crafted minimal case for the slice arithmetic, independent of
    any fixture. Catches off-by-one in (r0, c0, src_rows, src_cols)
    that the fixture comparison might mask if both implementations
    have the same bug.
    """
    src_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    src_b = torch.tensor([[10.0]], dtype=DTYPE)
    cowan_section = [src_a, src_b]

    add_entries = [
        # Place src_a at (0, 0) into a 4x4 matrix
        ADDEntry(matrix_idx=1, bra=1, ket=1, nbra=2, nket=2, coeff=1.0),
        # Place src_b at (3, 3) into the same matrix with coeff=5
        ADDEntry(matrix_idx=2, bra=4, ket=4, nbra=1, nket=1, coeff=5.0),
        # Place src_a again at (1, 2) with coeff=-1 (overlapping case)
        ADDEntry(matrix_idx=1, bra=2, ket=3, nbra=2, nket=2, coeff=-1.0),
    ]

    mat = assemble_matrix_from_adds(add_entries, cowan_section, 4, 4)

    expected = torch.zeros(4, 4, dtype=DTYPE)
    expected[0:2, 0:2] += 1.0 * src_a
    expected[3:4, 3:4] += 5.0 * src_b
    expected[1:3, 2:4] += -1.0 * src_a

    assert torch.allclose(mat, expected), f"\ngot:\n{mat}\nexpected:\n{expected}"


@pytest.mark.phase2
def test_out_of_bounds_add_entry_silently_clipped():
    """
    The legacy implementation tolerated ADD entries whose target
    extends past the matrix dimensions (per-element bounds check).
    The vectorized version must preserve this — clip the slice rather
    than raise.
    """
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                       dtype=DTYPE)
    cowan_section = [src]

    # 3x3 source placed at (1, 1) in a 3x3 matrix → only the top-left
    # 2x2 of src lands inside the destination.
    add_entries = [
        ADDEntry(matrix_idx=1, bra=2, ket=2, nbra=3, nket=3, coeff=1.0),
    ]

    new = assemble_matrix_from_adds(add_entries, cowan_section, 3, 3)
    old = _assemble_matrix_from_adds_legacy(add_entries, cowan_section, 3, 3)
    assert torch.equal(new, old)

    # Sanity: mat[1:3, 1:3] = src[0:2, 0:2]
    expected = torch.zeros(3, 3, dtype=DTYPE)
    expected[1:3, 1:3] = src[0:2, 0:2]
    assert torch.allclose(new, expected)


@pytest.mark.phase2
def test_empty_add_entries_returns_zeros():
    """No ADD entries → zero matrix."""
    mat = assemble_matrix_from_adds([], [torch.zeros(2, 2)], 4, 4)
    assert torch.equal(mat, torch.zeros(4, 4, dtype=DTYPE))


@pytest.mark.phase2
def test_invalid_matrix_idx_silently_skipped():
    """
    matrix_idx out of range → entry is silently skipped (legacy behavior).
    """
    src = torch.ones(2, 2, dtype=DTYPE)
    cowan_section = [src]
    add_entries = [
        ADDEntry(matrix_idx=99, bra=1, ket=1, nbra=2, nket=2, coeff=1.0),  # bad
        ADDEntry(matrix_idx=1, bra=1, ket=1, nbra=2, nket=2, coeff=1.0),   # good
    ]
    mat = assemble_matrix_from_adds(add_entries, cowan_section, 2, 2)
    assert torch.allclose(mat, src)
