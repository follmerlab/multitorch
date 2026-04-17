"""
multitorch: PyTorch implementation of multiplet X-ray spectroscopy simulations.

Implements core-hole X-ray spectroscopy (XAS, XES, RIXS, XMCD) via
crystal field and charge transfer multiplet theory, replacing the
Fortran ttmult/ttrcg/ttban suite with fully differentiable PyTorch tensors.

Usage:
    from multitorch import calcXAS
    x, y = calcXAS(element='Ni', valence='ii', sym='d4h', edge='l',
                   cf={'tendq': 1.0, 'ds': 0.0, 'dt': 0.01})
"""

# Public API
from multitorch.api.calc import (
    calcXAS, calcXAS_from_scratch, calcXES, calcRIXS, calcDOC,
    preload_fixture, calcXAS_cached, CachedFixture,
)
from multitorch.api.plot import getXAS, getXES

__version__ = "0.1.0.dev0"
