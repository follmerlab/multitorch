"""
Tests for batch atomic parameter scaling (Phase 2A step 1).

Validates that batch_scale_atomic_params produces correct results and
preserves per-sample autograd for parameter refinement workflows.
"""
import pytest
import torch

from multitorch.atomic.scaled_params import (
    scale_atomic_params,
    batch_scale_atomic_params,
)
from multitorch.atomic.parameter_fixtures import read_rcn31_out_params
from multitorch.api.calc import preload_fixture


@pytest.fixture
def ni_params():
    """Load Ni d8 atomic parameters."""
    cache = preload_fixture("Ni", "ii", "d4h")
    return cache.raw_params


def test_batch_scale_basic(ni_params):
    """Verify batch scaling produces correct shapes and values."""
    N = 10
    slater_vals = torch.linspace(0.6, 1.0, N)
    soc_vals = torch.linspace(0.8, 1.2, N)
    
    scaled_batch = batch_scale_atomic_params(ni_params, slater_vals, soc_vals)
    
    # Check that each Fk/Gk/ζ is now (N,) instead of scalar
    for key in scaled_batch.ground.fk.keys():
        assert scaled_batch.ground.fk[key].shape == (N,)
    
    for key in scaled_batch.ground.gk.keys():
        assert scaled_batch.ground.gk[key].shape == (N,)
    
    for key in scaled_batch.ground.zeta.keys():
        assert scaled_batch.ground.zeta[key].shape == (N,)


def test_batch_vs_sequential_parity(ni_params):
    """Verify batch results match sequential for each sample."""
    N = 5
    slater_vals = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    soc_vals = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Batch version
    scaled_batch = batch_scale_atomic_params(ni_params, slater_vals, soc_vals)
    
    # Sequential version
    for i in range(N):
        scaled_i = scale_atomic_params(
            ni_params, slater_scale=float(slater_vals[i]), soc_scale=float(soc_vals[i])
        )
        
        # Check each Fk value matches
        for key in scaled_batch.ground.fk.keys():
            batch_val_i = scaled_batch.ground.fk[key][i]
            seq_val_i = scaled_i.ground.fk[key]
            assert torch.allclose(batch_val_i, seq_val_i, atol=1e-10)
        
        # Check each Gk value matches
        for key in scaled_batch.ground.gk.keys():
            batch_val_i = scaled_batch.ground.gk[key][i]
            seq_val_i = scaled_i.ground.gk[key]
            assert torch.allclose(batch_val_i, seq_val_i, atol=1e-10)
        
        # Check each ζ value matches
        for key in scaled_batch.ground.zeta.keys():
            batch_val_i = scaled_batch.ground.zeta[key][i]
            seq_val_i = scaled_i.ground.zeta[key]
            assert torch.allclose(batch_val_i, seq_val_i, atol=1e-10)


def test_batch_autograd_preservation(ni_params):
    """Verify per-sample gradients are independent."""
    N = 3
    slater_vals = torch.tensor([0.7, 0.8, 0.9], requires_grad=True)
    soc_vals = torch.tensor([0.9, 1.0, 1.1], requires_grad=True)
    
    scaled_batch = batch_scale_atomic_params(ni_params, slater_vals, soc_vals)
    
    # Check that batch tensors require gradients
    for key in scaled_batch.ground.fk.keys():
        assert scaled_batch.ground.fk[key].requires_grad
    
    # Compute loss on one sample only
    loss = scaled_batch.ground.f("3D", "3D", 2)[1].sum()  # Only sample 1
    loss.backward()
    
    # Check gradient on slater_vals
    assert slater_vals.grad is not None
    # Only slater_vals[1] should have nonzero gradient (sample 1)
    assert slater_vals.grad[0] == 0.0  # Sample 0 not in loss
    assert slater_vals.grad[1] != 0.0  # Sample 1 in loss
    assert slater_vals.grad[2] == 0.0  # Sample 2 not in loss


def test_batch_shape_validation(ni_params):
    """Verify shape validation catches mismatches."""
    # Mismatched batch sizes
    slater_vals = torch.linspace(0.6, 1.0, 10)
    soc_vals = torch.linspace(0.8, 1.2, 5)  # Different size
    
    with pytest.raises(ValueError, match="Batch size mismatch"):
        batch_scale_atomic_params(ni_params, slater_vals, soc_vals)
    
    # 2D input (should be 1D)
    slater_2d = torch.randn(5, 2)
    soc_1d = torch.randn(5)
    
    with pytest.raises(ValueError, match="must be 1D"):
        batch_scale_atomic_params(ni_params, slater_2d, soc_1d)


def test_batch_dtype_consistency(ni_params):
    """Verify batch respects DTYPE (float64)."""
    from multitorch._constants import DTYPE
    
    # Input as float32
    slater_vals = torch.linspace(0.6, 1.0, 5, dtype=torch.float32)
    soc_vals = torch.linspace(0.8, 1.2, 5, dtype=torch.float32)
    
    scaled_batch = batch_scale_atomic_params(ni_params, slater_vals, soc_vals)
    
    # Output should be DTYPE (float64)
    for key in scaled_batch.ground.fk.keys():
        assert scaled_batch.ground.fk[key].dtype == DTYPE
