"""
Convenience wrappers matching pyctm.plot API.

getXAS, getXES, getRIXS mirror the pyctm interface for drop-in compatibility.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from multitorch.api.calc import calcXAS, calcXES, calcRIXS


def getXAS(
    ban_output_path: str,
    T: float = 80.0,
    beam_fwhm: float = 0.2,
    gamma1: float = 0.2,
    gamma2: float = 0.4,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get XAS spectrum from a .ban_out file (bootstrap mode)."""
    return calcXAS(
        element='', valence='', sym='', edge='',
        cf={}, T=T, beam_fwhm=beam_fwhm,
        gamma1=gamma1, gamma2=gamma2,
        ban_output_path=ban_output_path,
        **kwargs,
    )


def getXES(ban_output_path: str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get XES spectrum from a .ban_out file."""
    return getXAS(ban_output_path, **kwargs)
