"""
Deployment audit tests — behavioral checks that go beyond implementation.

TEST-01: Hermiticity of assembled Hamiltonians
TEST-04: d0/d10 edge cases (empty d-shell / full d-shell)
TEST-05: Zero crystal field → degeneracy within Oh irreps
TEST-06: Degenerate eigh backward stability (safe_eigh guard)
"""
import pytest
import torch

from multitorch._constants import DTYPE
from multitorch.hamiltonian.diagonalize import diagonalize, safe_eigh


# ────────────────────────────────────────────────────────────────
# TEST-01: Hermiticity of assembled Hamiltonians
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestHermiticity:
    """Assembled Hamiltonians must be real-symmetric (Hermitian)."""

    def test_diagonalize_output_reconstructs_symmetric(self):
        """V @ diag(λ) @ V^T must be symmetric for a random H."""
        n = 20
        A = torch.randn(n, n, dtype=DTYPE)
        H = 0.5 * (A + A.T)
        eigvals, eigvecs = diagonalize(H)
        H_rec = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        # Reconstruction is symmetric
        assert torch.allclose(H_rec, H_rec.T, atol=1e-10), \
            f"Reconstructed H is not symmetric: max asymmetry = {(H_rec - H_rec.T).abs().max():.2e}"
        # Reconstruction matches original
        torch.testing.assert_close(H_rec, H, atol=1e-10, rtol=0)

    def test_random_hamiltonian_eigenvalues_real(self):
        """All eigenvalues of a symmetric matrix must be real (no imaginary part)."""
        n = 50
        A = torch.randn(n, n, dtype=DTYPE)
        H = 0.5 * (A + A.T)
        eigvals, _ = diagonalize(H)
        assert eigvals.is_floating_point(), "Eigenvalues should be real-valued"
        assert not eigvals.isnan().any(), "No NaN eigenvalues"
        assert not eigvals.isinf().any(), "No Inf eigenvalues"


# ────────────────────────────────────────────────────────────────
# TEST-05: Zero crystal field → degeneracy within irreps
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestZeroCrystalField:
    """With zero CF, the Hamiltonian becomes block-diagonal with
    degenerate eigenvalues within each J-multiplet."""

    def test_zero_cf_produces_degenerate_eigenvalues(self):
        """A diagonal Hamiltonian with repeated entries should give
        degenerate eigenvalues."""
        # Simulate a zero-CF Hamiltonian: 3-fold degenerate block
        H = torch.zeros(6, 6, dtype=DTYPE)
        H[0, 0] = H[1, 1] = H[2, 2] = -1.0  # t2g block
        H[3, 3] = H[4, 4] = H[5, 5] = 0.5   # eg block
        eigvals, _ = diagonalize(H)
        # Should get three -1.0 and three 0.5
        assert torch.allclose(eigvals[:3], torch.full((3,), -1.0, dtype=DTYPE), atol=1e-12)
        assert torch.allclose(eigvals[3:], torch.full((3,), 0.5, dtype=DTYPE), atol=1e-12)


# ────────────────────────────────────────────────────────────────
# TEST-06: Degenerate eigh backward stability (safe_eigh guard)
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestSafeEigh:
    """safe_eigh must produce finite gradients even at exact degeneracies."""

    def test_degenerate_eigenvalues_no_nan_grad(self):
        """Backward through safe_eigh on a matrix with exact degeneracies
        must not produce NaN or Inf gradients."""
        n = 6
        # Exactly degenerate: 3-fold at 0.0, 3-fold at 1.0
        diag_vals = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=DTYPE)
        H = torch.diag(diag_vals).requires_grad_(True)

        eigvals, eigvecs = safe_eigh(H)

        # Backward on sum of eigenvalues (simple scalar loss)
        loss = eigvals.sum()
        loss.backward()

        assert H.grad is not None, "Gradient must exist"
        assert not H.grad.isnan().any(), \
            f"NaN in gradient! grad =\n{H.grad}"
        assert not H.grad.isinf().any(), \
            f"Inf in gradient! grad =\n{H.grad}"

    def test_safe_eigh_no_perturbation_without_grad(self):
        """Without requires_grad, safe_eigh should give exact results
        (no perturbation applied)."""
        H = torch.diag(torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE))
        eigvals, _ = safe_eigh(H)
        # Exact: [1.0, 1.0, 2.0]
        torch.testing.assert_close(
            eigvals,
            torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE),
            atol=1e-14, rtol=0,
        )

    def test_safe_eigh_perturbation_is_tiny(self):
        """With requires_grad, the perturbation should be negligible
        (eigenvalues should still match to ~1e-10)."""
        H = torch.diag(torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE)).requires_grad_(True)
        eigvals, _ = safe_eigh(H)
        torch.testing.assert_close(
            eigvals,
            torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE),
            atol=1e-10, rtol=0,
        )

    def test_gradcheck_safe_eigh(self):
        """torch.autograd.gradcheck on safe_eigh with symmetric parameterization.

        We parameterize the input as L (lower triangle) and construct
        H = L + L^T - diag(L) so the input to safe_eigh is always
        symmetric regardless of perturbation direction.
        """
        L = torch.tensor([
            [3.0, 0.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.1, 0.3, 1.0],
        ], dtype=DTYPE, requires_grad=True)

        def fn(lower):
            h = lower + lower.T - torch.diag(lower.diag())
            eigvals, _ = safe_eigh(h)
            return eigvals

        assert torch.autograd.gradcheck(fn, (L,), eps=1e-6, atol=1e-4), \
            "gradcheck failed for safe_eigh"


# ────────────────────────────────────────────────────────────────
# TEST-04: d0/d10 edge cases
# ────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestEdgeCases:
    """Edge cases: empty and full d-shell should produce trivial spectra."""

    def test_1x1_hamiltonian(self):
        """A 1×1 Hamiltonian (single state, like d0 or d10) should return
        that state as the sole eigenvalue."""
        H = torch.tensor([[5.0]], dtype=DTYPE)
        eigvals, eigvecs = diagonalize(H)
        assert eigvals.shape == (1,)
        assert eigvecs.shape == (1, 1)
        torch.testing.assert_close(eigvals, torch.tensor([5.0], dtype=DTYPE))
        torch.testing.assert_close(eigvecs, torch.tensor([[1.0]], dtype=DTYPE))

    def test_empty_matrix(self):
        """A 0×0 Hamiltonian (no states) should work without error."""
        H = torch.zeros(0, 0, dtype=DTYPE)
        eigvals, eigvecs = diagonalize(H)
        assert eigvals.shape == (0,)
        assert eigvecs.shape == (0, 0)
