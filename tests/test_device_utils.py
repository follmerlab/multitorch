"""
Tests for device selection utilities.

Verifies that get_optimal_device() correctly selects CPU/GPU based on
operation characteristics and problem size.
"""
import pytest
import torch

from multitorch.device_utils import (
    get_optimal_device,
    suggest_device_for_xas,
    suggest_device_for_rixs,
)


class TestGetOptimalDevice:
    """Test automatic device selection based on operation type and size."""

    def test_rixs_prefers_gpu_if_available(self):
        """RIXS should always prefer GPU (45× speedup measured)."""
        device = get_optimal_device(operation='rixs')
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'

    def test_small_xas_prefers_cpu(self):
        """Small L-edge XAS (typical 3d TM) should use CPU."""
        device = get_optimal_device(operation='xas', matrix_dim=17)
        assert device == 'cpu'

    def test_large_xas_prefers_gpu_if_available(self):
        """Large rare earth XAS should use GPU if available."""
        device = get_optimal_device(operation='xas', matrix_dim=1000)
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'

    def test_small_eigh_prefers_cpu(self):
        """Small matrix diagonalization should use CPU (kernel overhead)."""
        device = get_optimal_device(operation='eigh', matrix_dim=200)
        assert device == 'cpu'

    def test_large_eigh_prefers_gpu_if_available(self):
        """Large matrix diagonalization should use GPU if available."""
        device = get_optimal_device(operation='eigh', matrix_dim=800)
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'

    def test_small_broaden_prefers_cpu(self):
        """Small broadening (few sticks) should use CPU."""
        device = get_optimal_device(operation='broaden', n_sticks=100)
        assert device == 'cpu'

    def test_large_broaden_prefers_gpu_if_available(self):
        """Large broadening (many sticks) should use GPU if available."""
        device = get_optimal_device(operation='broaden', n_sticks=5000)
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'

    def test_force_device_overrides_automatic(self):
        """Explicit device parameter should override automatic selection."""
        device = get_optimal_device(operation='rixs', force_device='cpu')
        assert device == 'cpu'

        device = get_optimal_device(operation='xas', matrix_dim=17, force_device='cuda:0')
        assert device == 'cuda:0'


class TestSuggestDeviceForXAS:
    """Test XAS-specific device suggestions."""

    def test_default_is_cpu(self):
        """XAS should default to CPU for typical use cases."""
        device = suggest_device_for_xas()
        assert device == 'cpu'

    def test_3d_tm_uses_cpu(self):
        """3d transition metals should use CPU."""
        device = suggest_device_for_xas(element='Ni', valence='ii')
        assert device == 'cpu'

        device = suggest_device_for_xas(element='Fe', valence='iii')
        assert device == 'cpu'

    def test_force_device_overrides(self):
        """Explicit device should override default."""
        device = suggest_device_for_xas(force_device='cuda')
        assert device == 'cuda'


class TestSuggestDeviceForRIXS:
    """Test RIXS-specific device suggestions."""

    def test_prefers_gpu_if_available(self):
        """RIXS should prefer GPU (45× speedup)."""
        device = suggest_device_for_rixs()
        if torch.cuda.is_available():
            assert device == 'cuda'
        else:
            assert device == 'cpu'

    def test_force_device_overrides(self):
        """Explicit device should override default."""
        device = suggest_device_for_rixs(force_device='cpu')
        assert device == 'cpu'
