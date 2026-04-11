"""
Phase 2 tests: Hamiltonian diagonalization.

Tests:
  1. torch.linalg.eigh wrapper correctness
  2. Known 2×2 and 3×3 Hamiltonians with exact eigenvalues
  3. Symmetrization of nearly-symmetric matrices
  4. Eigenvector orthonormality
  5. Crystal field branch coefficient values match Butler-Wybourne tables
  6. Transition matrix assembly for a simple case
"""
import pytest
import torch
import math


@pytest.mark.phase2
def test_diagonalize_2x2():
    """
    Known 2×2 symmetric matrix:
      H = [[1, 2], [2, 1]]
    Eigenvalues: 3, -1 (sorted: -1, 3)
    """
    from multitorch.hamiltonian.diagonalize import diagonalize
    H = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    eigvals, eigvecs = diagonalize(H)
    torch.testing.assert_close(
        eigvals,
        torch.tensor([-1.0, 3.0], dtype=torch.float64),
        atol=1e-12, rtol=0,
    )


@pytest.mark.phase2
def test_diagonalize_3x3_hamiltonian():
    """
    Simple 3×3 crystal field Hamiltonian for t2g block.
    H = diag(-4, -4, -4) + hopping: eigenvalues are -4 (3-fold degenerate)
    """
    from multitorch.hamiltonian.diagonalize import diagonalize
    H = -4.0 * torch.eye(3, dtype=torch.float64)
    eigvals, eigvecs = diagonalize(H)
    torch.testing.assert_close(
        eigvals,
        torch.full((3,), -4.0, dtype=torch.float64),
        atol=1e-12, rtol=0,
    )


@pytest.mark.phase2
def test_diagonalize_eigenvectors_orthonormal():
    """Eigenvectors from diagonalize() should be orthonormal."""
    from multitorch.hamiltonian.diagonalize import diagonalize
    H = torch.tensor([
        [3.0, 1.0, 0.5],
        [1.0, 2.0, 0.8],
        [0.5, 0.8, 1.0],
    ], dtype=torch.float64)
    _, eigvecs = diagonalize(H)
    # V^T V should be I
    VTV = eigvecs.T @ eigvecs
    torch.testing.assert_close(VTV, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.phase2
def test_diagonalize_reconstruction():
    """V diag(λ) V^T should recover H."""
    from multitorch.hamiltonian.diagonalize import diagonalize
    H = torch.tensor([
        [4.0, 2.0, 1.0],
        [2.0, 3.0, 1.5],
        [1.0, 1.5, 2.0],
    ], dtype=torch.float64)
    eigvals, eigvecs = diagonalize(H)
    H_rec = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    torch.testing.assert_close(H_rec, H, atol=1e-12, rtol=0)


@pytest.mark.phase2
def test_diagonalize_symmetrizes_input():
    """Slightly asymmetric matrix should be symmetrized and diagonalized."""
    from multitorch.hamiltonian.diagonalize import diagonalize
    H = torch.tensor([[1.0, 2.0 + 1e-14], [2.0, 1.0]], dtype=torch.float64)
    eigvals, _ = diagonalize(H)
    torch.testing.assert_close(
        eigvals,
        torch.tensor([-1.0, 3.0], dtype=torch.float64),
        atol=1e-10, rtol=0,
    )


@pytest.mark.phase2
def test_crystal_field_d4h_branch_values():
    """
    Butler-Wybourne branch coefficients for D4h symmetry:
      X40 = 6*sqrt(30) * (10Dq/10)  for 10Dq = 1.0 eV
      X420 = -5/2 * sqrt(42) * dt    for dt = 0.01 eV
      X220 = -sqrt(70) * ds           for ds = 0.0 eV

    Test against the exact algebraic values from write_RAC.py.
    """
    from multitorch.hamiltonian.crystal_field import get_cf_branch_values
    cf = get_cf_branch_values('d4h', tendq=1.0, ds=0.0, dt=0.01)
    # X40 = 6*sqrt(30) * ((1.0 - 35*0.01/6) / 10)
    tendq_eff = 1.0 - 35.0 * 0.01 / 6.0
    expected_X40 = 6.0 * math.sqrt(30.0) * tendq_eff / 10.0
    assert abs(cf['X40'] - expected_X40) < 1e-10, f"X40 = {cf['X40']:.6f}, expected {expected_X40:.6f}"


@pytest.mark.phase2
def test_crystal_field_oh_branch_values():
    """For Oh symmetry, only 10Dq contributes."""
    from multitorch.hamiltonian.crystal_field import get_cf_branch_values
    cf = get_cf_branch_values('oh', tendq=1.0)
    expected_X40 = 6.0 * math.sqrt(30.0) * 1.0 / 10.0
    assert abs(cf['X40'] - expected_X40) < 1e-10


@pytest.mark.phase2
def test_transition_matrix_simple():
    """
    Simple 2×2 transition matrix test.
    T_rme = [[1, 0], [0, 1]] (identity, pure diagonal)
    U_gs = I, U_fs = I → M = T²  = I
    """
    from multitorch.hamiltonian.transitions import build_transition_matrix
    T = torch.eye(2, dtype=torch.float64)
    U_gs = torch.eye(2, dtype=torch.float64)
    U_fs = torch.eye(2, dtype=torch.float64)
    M = build_transition_matrix(T, U_gs, U_fs)
    # M = (I^T @ I @ I)^2 = I^2 = I
    torch.testing.assert_close(M, torch.eye(2, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.phase2
def test_transition_matrix_rotated():
    """
    With rotated eigenvectors, transition matrix should change.
    """
    from multitorch.hamiltonian.transitions import build_transition_matrix
    import math
    T = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    # Rotate by 45 degrees
    angle = math.pi / 4
    U = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=torch.float64)
    M = build_transition_matrix(T, U, U)
    # M = (U^T @ T @ U)^2 = [[0.25, 0.25], [0.25, 0.25]]
    expected = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float64)
    torch.testing.assert_close(M, expected, atol=1e-12, rtol=0)
