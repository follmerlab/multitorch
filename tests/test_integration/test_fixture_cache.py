"""Tests for CachedFixture / calcXAS_cached fast-path API."""
import torch
import pytest

from multitorch.api.calc import preload_fixture, calcXAS_cached, calcXAS


def test_cached_matches_uncached():
    """calcXAS_cached produces identical output to calcXAS."""
    cache = preload_fixture("Ni", "ii", "oh")

    cf = {'tendq': 1.0}
    x_cached, y_cached = calcXAS_cached(cache, cf=cf, slater=0.8, soc=1.0)
    x_direct, y_direct = calcXAS(
        "Ni", "ii", "oh", "l", cf=cf, slater=0.8, soc=1.0,
    )

    assert torch.allclose(x_cached, x_direct, atol=1e-10)
    assert torch.allclose(y_cached, y_direct, atol=1e-10)


def test_cached_different_params_differ():
    """Different cf values produce different spectra from cache."""
    cache = preload_fixture("Ni", "ii", "oh")

    _, y1 = calcXAS_cached(cache, cf={'tendq': 0.5}, slater=0.8)
    _, y2 = calcXAS_cached(cache, cf={'tendq': 2.0}, slater=0.8)

    # Different 10Dq values should produce different spectra
    assert not torch.allclose(y1, y2, atol=1e-6)


def test_cached_autograd():
    """Autograd flows through slater_scale in cached path."""
    cache = preload_fixture("Ni", "ii", "oh")

    slater = torch.tensor(0.8, requires_grad=True, dtype=torch.float64)
    x, y = calcXAS_cached(cache, cf={}, slater=slater, soc=1.0)

    loss = y.sum()
    grad = torch.autograd.grad(loss, slater)
    assert torch.isfinite(grad[0]).all()
    assert grad[0].abs() > 1e-6


def test_cached_sweep_consistency():
    """A sweep over the cache produces the same results as direct calls."""
    cache = preload_fixture("Ni", "ii", "oh")

    for tendq_val in [0.5, 1.0, 1.5]:
        cf = {'tendq': tendq_val}
        x_c, y_c = calcXAS_cached(cache, cf=cf, slater=0.8)
        x_d, y_d = calcXAS("Ni", "ii", "oh", "l", cf=cf, slater=0.8)
        assert torch.allclose(y_c, y_d, atol=1e-10), (
            f"Mismatch at tendq={tendq_val}"
        )
