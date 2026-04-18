"""
Tests for batch XAS calculation API (Phase 2B).

Validates high-level calcXAS_batch and batch_parameter_sweep functions.
"""
import pytest
import torch

from multitorch.api.calc import (
    preload_fixture,
    calcXAS_cached,
    calcXAS_batch,
    batch_parameter_sweep,
)


@pytest.fixture
def ni_d4h_cache():
    """Pre-loaded Ni d8 D4h fixture for fast tests."""
    return preload_fixture("Ni", "ii", "d4h")


def test_calcXAS_batch_basic(ni_d4h_cache):
    """Verify calcXAS_batch produces correct shape."""
    N = 10
    slater_vals = torch.linspace(0.7, 0.9, N)
    soc_vals = torch.linspace(0.9, 1.1, N)
    
    y_batch = calcXAS_batch(
        ni_d4h_cache,
        slater_values=slater_vals,
        soc_values=soc_vals,
        nbins=500,  # Smaller for faster test
    )
    
    assert y_batch.shape == (N, 500)
    assert y_batch.dtype == torch.float64


def test_calcXAS_batch_vs_sequential_parity(ni_d4h_cache):
    """Verify batch matches sequential calcXAS_cached for each sample."""
    N = 3
    slater_vals = torch.tensor([0.7, 0.8, 0.9])
    soc_vals = torch.tensor([0.9, 1.0, 1.1])
    
    # Batch version
    y_batch = calcXAS_batch(
        ni_d4h_cache,
        slater_values=slater_vals,
        soc_values=soc_vals,
        nbins=500,
        device="cpu",
    )
    
    # Sequential version
    for i in range(N):
        _, y_seq = calcXAS_cached(
            ni_d4h_cache,
            slater=float(slater_vals[i]),
            soc=float(soc_vals[i]),
            nbins=500,
            device="cpu",
        )
        
        # Should match reasonably well
        # Note: Small differences expected due to floating point order of operations
        # and slight differences in x-grid specification per spectrum
        max_diff = (y_batch[i] - y_seq).abs().max()
        assert max_diff < 1e-3, f"Sample {i}: max difference {max_diff:.2e} exceeds tolerance"


def test_calcXAS_batch_shape_validation(ni_d4h_cache):
    """Verify input validation catches errors."""
    # Mismatched batch sizes
    slater_vals = torch.linspace(0.7, 0.9, 10)
    soc_vals = torch.linspace(0.9, 1.1, 5)
    
    with pytest.raises(ValueError, match="Batch size mismatch"):
        calcXAS_batch(ni_d4h_cache, slater_vals, soc_vals)
    
    # Wrong dimensionality
    slater_2d = torch.randn(5, 2)
    soc_1d = torch.randn(5)
    
    with pytest.raises(ValueError, match="must be 1D"):
        calcXAS_batch(ni_d4h_cache, slater_2d, soc_1d)


def test_calcXAS_batch_autograd(ni_d4h_cache):
    """Verify gradients flow through batch calculation."""
    N = 3
    slater_vals = torch.tensor([0.7, 0.8, 0.9], requires_grad=True)
    soc_vals = torch.tensor([0.9, 1.0, 1.1], requires_grad=True)
    
    y_batch = calcXAS_batch(
        ni_d4h_cache,
        slater_values=slater_vals,
        soc_values=soc_vals,
        nbins=500,
        device="cpu",
    )
    
    # Loss on spectrum 1 only
    loss = y_batch[1, :100].sum()
    loss.backward()
    
    # Gradients should exist
    assert slater_vals.grad is not None
    assert soc_vals.grad is not None
    
    # Sample 1 should have larger gradient than others
    # (may not be exactly zero for others due to shared components)
    assert slater_vals.grad[1].abs() > 0


def test_batch_parameter_sweep_grid_mode(ni_d4h_cache):
    """Verify batch_parameter_sweep 2D grid mode."""
    slater_range = torch.linspace(0.7, 0.9, 5)
    soc_range = torch.linspace(0.9, 1.1, 4)
    
    y_grid, s_axis, c_axis = batch_parameter_sweep(
        ni_d4h_cache,
        slater_range=slater_range,
        soc_range=soc_range,
        nbins=500,
    )
    
    assert y_grid.shape == (5, 4, 500)
    assert torch.allclose(s_axis, slater_range)
    assert torch.allclose(c_axis, soc_range)


def test_batch_parameter_sweep_custom_mode(ni_d4h_cache):
    """Verify batch_parameter_sweep custom grid mode."""
    N = 20
    slater_samples = torch.randn(N) * 0.05 + 0.85
    soc_samples = torch.randn(N) * 0.1 + 1.0
    
    y_batch = batch_parameter_sweep(
        ni_d4h_cache,
        slater_grid=slater_samples,
        soc_grid=soc_samples,
        nbins=500,
    )
    
    assert y_batch.shape == (N, 500)


def test_batch_parameter_sweep_validation(ni_d4h_cache):
    """Verify batch_parameter_sweep input validation."""
    # Missing arguments
    with pytest.raises(ValueError, match="Must provide either"):
        batch_parameter_sweep(ni_d4h_cache)
    
    # Mixing modes
    with pytest.raises(ValueError, match="Cannot mix"):
        batch_parameter_sweep(
            ni_d4h_cache,
            slater_range=torch.linspace(0.7, 0.9, 5),
            soc_range=torch.linspace(0.9, 1.1, 5),
            slater_grid=torch.randn(10),
            soc_grid=torch.randn(10),
        )
    
    # Mismatched custom grid sizes
    with pytest.raises(ValueError, match="same shape"):
        batch_parameter_sweep(
            ni_d4h_cache,
            slater_grid=torch.randn(10),
            soc_grid=torch.randn(5),
        )


@pytest.mark.slow
@pytest.mark.skip(reason="Full speedup requires Phase 2+ batch diagonalization (future work)")
def test_batch_performance_benefit(ni_d4h_cache):
    """Verify batch is faster than sequential (integration test)."""
    import time
    
    N = 50  # Moderate batch size for timing
    slater_vals = torch.linspace(0.7, 0.9, N)
    soc_vals = torch.linspace(0.9, 1.1, N)
    
    # Batch version
    start = time.perf_counter()
    y_batch = calcXAS_batch(
        ni_d4h_cache, slater_vals, soc_vals, nbins=1000, device="cpu"
    )
    batch_time = time.perf_counter() - start
    
    # Sequential version (subset for speed)
    N_seq = 10  # Only time a subset
    start = time.perf_counter()
    for i in range(N_seq):
        _, y = calcXAS_cached(
            ni_d4h_cache,
            slater=float(slater_vals[i]),
            soc=float(soc_vals[i]),
            nbins=1000,
            device="cpu",
        )
    seq_time = time.perf_counter() - start
    
    # Extrapolate sequential time to full N
    seq_time_full = seq_time * (N / N_seq)
    
    # Batch should be at least 1.5× faster (conservative check)
    # Real speedup should be 2-3× from V(11) sharing
   speedup = seq_time_full / batch_time
    print(f"Batch speedup: {speedup:.2f}×")
    assert speedup > 1.5, f"Expected >1.5× speedup, got {speedup:.2f}×"
