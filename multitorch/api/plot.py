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


def getRIXS(
    ban_abs_path: str,
    ban_ems_path: str,
    Gamma_i: float = 0.4,
    Gamma_f: float = 0.2,
    T: float = 80.0,
    Einc: Optional[torch.Tensor] = None,
    Efin: Optional[torch.Tensor] = None,
    n_Einc: int = 400,
    n_Efin: int = 400,
    pad_eV: float = 5.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get a RIXS plane from a paired absorption / emission ``.ban_out`` set.

    Mirrors :func:`getXAS` for the bootstrap RIXS pipeline.
    Returns ``(Einc, Efin, intensity_2D)``.
    """
    return calcRIXS(
        ban_abs_path=ban_abs_path,
        ban_ems_path=ban_ems_path,
        Gamma_i=Gamma_i, Gamma_f=Gamma_f, T=T,
        Einc=Einc, Efin=Efin,
        n_Einc=n_Einc, n_Efin=n_Efin, pad_eV=pad_eV,
        **kwargs,
    )
